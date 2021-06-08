
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import length2mask
import utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

class FixedSizeMemory(nn.Module):
    """
    Transfer the variable_length memory into fixed_size memory.
    """
    def __init__(self, input_dim, mem_size):
        super(FixedSizeMemory, self).__init__()
        self.mem_size = mem_size
        self.f = nn.Linear(input_dim, mem_size)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, ctx, length=None):
        """
        output the fix_length memory
        :param ctx: (batch_size, max_len, input_dim)
        :param length: (batch_size)
        :return:
        """
        attn = self.f(ctx)                          # (batch_size, max_len, mem_size)
        if length is not None:
            mask = utils.length2mask(length).unsqueeze(-1).expand(-1, -1, self.mem_size)
            # print(attn.size())
            # print(mask.size())
            attn.masked_fill_(mask, float('-inf'))
        attn = F.softmax(attn, 1, _stacklevel=5)    # (batch_size, max_len, mem_size)
        attn = attn.transpose(1, 2)                 # (batch_size, mem_size, max_len)
        memory = torch.bmm(attn, ctx)               # (batch_size, mem_size, rnn_dim)
        memory = self.drop(memory)
        return memory


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        # Change the input_size to 1024 (elmo)
        input_size = 1024 if args.elmo else embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        if args.fixed_size_ctx > 0:
            self.memory = FixedSizeMemory(self.hidden_size * self.num_directions, args.fixed_size_ctx)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

        # ELMO parameters
        self.elmo_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        if args.elmo:
            embeds = (inputs * F.softmax(self.elmo_weight).view(1, -1, 1, 1)).sum(1) * self.gamma
        else:
            embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.fixed_size_ctx > 0:
            ctx = self.memory(ctx, lengths)

        if args.sub_out == "max":
            # mask = length2mask(lengths)
            # ctx = ctx.masked_fill_(mask, float("-inf"))
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim, method='bilinear'):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, pre_alpha=None, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        #attn = F.softmax(attn, -1)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        if args.traditional_drop:
            self.feature_fc = nn.Linear(self.feature_size - args.angle_feat_size, hidden_size)
            feature_size = hidden_size + args.angle_feat_size
        if args.feat_batch_norm:
            self.fbn = nn.BatchNorm1d(self.feature_size - args.angle_feat_size, affine=False)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        if args.denoise:
            self.gate_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                self.drop,
                nn.Linear(hidden_size, 1),
            )
            # self.gate_layer.

    def forward(self, action, feature, cand_feat, cand_len,
                prev_feat, h_0, prev_h1, c_0, pre_alpha, ctx, ctx_mask=None, already_dropfeat=False):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 4
        feature: batch x 36 x (feature_size + 4)
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            if args.traditional_drop:
                x = F.relu(self.feature_fc(feature[..., :-args.angle_feat_size]))
                x = self.drop3(x)
                # x = F.relu(self.feature_fc(self.drop3(feature[..., :-args.angle_feat_size])))
                feature = torch.cat((x, feature[..., -args.angle_feat_size:]), -1)
            else:
                feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        # prev_h1 = prev_h1.detach()
        prev_h1_drop = self.drop(prev_h1)

        # prev_h1, prev_h1_drop = prev_h1
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        # concat_input = self.drop(concat_input)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, pre_alpha, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            if args.traditional_drop:
                x = F.relu(self.feature_fc(cand_feat[..., :-args.angle_feat_size]))
                x = self.drop3(x)
                # x = F.relu(self.feature_fc(self.drop3(cand_feat[..., :-args.angle_feat_size])))
                cand_feat = torch.cat((x, cand_feat[..., -args.angle_feat_size:]), -1)
            else:
                cand_feat[..., :-args.angle_feat_size] = self.drop3(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, alpha, logit, h_tilde#, h_tilde_drop)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class PackSpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in PACK speaker encoder!!")

        self.lstm = nn.LSTM(feature_size * 2, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, first_feat, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """

        # Sort the sequence
        lengths = torch.LongTensor(lengths)
        lengths, perm_idx = lengths.sort(descending=True)
        _, reorder = perm_idx.sort()
        action_embeds, feature, first_feat = action_embeds[perm_idx], feature[perm_idx], first_feat[perm_idx]

        # Dropout Action Embedding
        if not already_dropfeat:
            action_embeds[..., :-4] = self.drop3(action_embeds[..., :-4])            # Do not dropout the spatial features

        # Embed the action_embeds and the first_feat
        action_embeds = torch.cat(
            (action_embeds,
             torch.cat((first_feat.view(-1, 1, self.feature_size), action_embeds), 1)[:, :-1, :]
             ),
            -1
        )      # [b, l, f] + ([b, 1, f], [b, l, f] --> [b, l+1, f] --> [b, l, f]) --> [b, l, 2f]

        # LSTM & Dropout
        action_embeds = nn.utils.rnn.pack_padded_sequence(action_embeds, lengths, batch_first=True)
        ctx, _ = self.lstm(action_embeds)
        ctx, _ = nn.utils.rnn.pad_packed_sequence(ctx, batch_first=True)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-4] = self.drop3(feature[..., :-4])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(-1, 36, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # LSTM & Dropout
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.post_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.drop(x)

        # Reorder the context according to the batch
        x = x[reorder]

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class FeedForwardSpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        """

        :param words:       [batch_size, inst_len]
        :param ctx:         [batch_size, act_len, rnn_dim]
        :param ctx_mask:    [batch_size, act_len]
        :param h0:          [batch_size, rnn_dim]
        :param c0:          [batch_size, rnn_dim]
        :return:
        """

        # Word Emb
        words = words.t()                   # (b, l) --> (l, b)
        embeds = self.embedding(words)      # (l, b) --> (l, b, emb_size)
        embeds = self.drop(embeds)

        # Split the embeds to a list
        embeds = torch.split(embeds, 1, dim=0)     # (l, b, e) --> [(1, b, e), ..., (1, b, e)]

        # LSTM
        logits = []
        h0, c0 = h0.squeeze(), c0.squeeze()
        for embed in embeds:
            h1, c1 = self.lstm(embed.squeeze(), (h0, c0))                   # (b, dim), (b, dim)
            h1 = self.drop(h1)

            h_tilde, _ = self.attention_layer(h1, ctx, mask=ctx_mask)       # Attention
            h_tilde_drop = self.drop(h_tilde)

            logits.append(self.projection(h_tilde_drop))                    # Predict word logit

            h0, c0 = h_tilde, c1

        logits = torch.stack(logits, 1).contiguous()

        return logits, h0.unsqueeze(0), c0.unsqueeze(0)

class BidirectionalArbiter(nn.Module):
    def __init__(self, vocab_size, feature_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.num_layers = 1
        self.num_directions = 1
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout_ratio)
        self.drop3 = nn.Dropout(0.3)

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.inst_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.feat_lstm = nn.LSTM(feature_size * 2, hidden_size, batch_first=True)

        self.feat2inst_att = SoftDotAttention(hidden_size, hidden_size)
        self.inst2feat_att = SoftDotAttention(hidden_size, hidden_size)

        self.feat_modeling = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.inst_modeling = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            self.drop,
            nn.Linear(hidden_size, 1)
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, img_feats, can_feats, feat_mask, insts, inst_mask):
        """

        :param img_feats: B x SeqLen x 36 x (feature_size + 4)
        :param can_feats: B x SeqLen x (feature_size + 4)
        :param feat_mask: B x SeqLen
        :param insts:     B x InstLen
        :param inst_mask: B x InstLen
        :return:
        """

        # Encode the Text
        inst_emb = self.embedding(insts)                        # B x InstLen x Emb, word emb
        inst_emb = self.drop(inst_emb)
        inst_ctx, _ = self.inst_lstm(inst_emb, self.init_state(inst_emb))        # B x InstLen x rnn_dim
        inst_ctx = self.drop(inst_ctx)

        # Encode the Image
        attn_feat = img_feats.mean(dim=2)                       # B x SeqLen x 36 x f --> B x SeqLen x f
        concat_input = torch.cat((can_feats, attn_feat), 2)     # B x SeqLen x 2f
        feat_emb = self.drop(concat_input)
        feat_ctx, _ = self.feat_lstm(feat_emb, self.init_state(feat_emb))             # B x SeqLen x rnn_dim
        feat_ctx = self.drop(feat_ctx)

        # Text to Image Attention
        bs, inst_len, _ = inst_ctx.size()
        att_inst, _ = self.inst2feat_att(inst_ctx.contiguous().view(-1, self.hidden_size),           # (B*InstLen) * rnn_dim
                                         utils.tile_batch(feat_ctx, inst_len),          # (B*InstLen) * SeqLen * rnn_dim
                                         mask=utils.tile_batch(feat_mask, inst_len),    # (B*InstLen) * SeqLen
                                         output_tilde=True, output_prob=True
                                         )
        att_inst = att_inst.view_as(inst_ctx)

        # Image to Text Attention
        bs, feat_len, _ = feat_ctx.size()
        att_feat, _ = self.feat2inst_att(feat_ctx.contiguous().view(-1, self.hidden_size),           # (B*SeqLen) * rnn_dim
                                         utils.tile_batch(inst_ctx, feat_len),          # (B*SeqLen) * InstLen * rnn_dim
                                         mask=utils.tile_batch(inst_mask, feat_len),    # (B*SeqLen) * InstLen
                                         output_tilde=True, output_prob=True
                                         )
        att_feat = att_feat.view_as(feat_ctx)

        # Modeling layers
        postatt_inst, _ = self.inst_modeling(att_inst, self.init_state(att_inst))
        postatt_feat, _ = self.feat_modeling(att_feat, self.init_state(att_feat))

        # postatt_inst = self.drop(postatt_inst)
        # postatt_feat = self.drop(postatt_feat)

        # Max pooling Layer
        max_inst, _ = postatt_inst.clone().masked_fill_(inst_mask.unsqueeze(-1), float('-inf')).max(1)
        max_feat, _ = postatt_feat.clone().masked_fill_(feat_mask.unsqueeze(-1), float('-inf')).max(1)
        # max_inst, _ = postatt_inst.max(1)
        # max_feat, _ = postatt_feat.max(1)


        # Combined Feature
        x = torch.cat([max_inst, max_feat], 1)

        # Image Shortcut
        # x, _ = feat_ctx.clone().masked_fill(feat_mask.unsqueeze(-1), float('-inf')).max(1)

        # Text Shortcut
        # x, _ = inst_ctx.clone().masked_fill(inst_mask.unsqueeze(-1), float('-inf')).max(1)

        # Projection Layer
        logit = self.projection(x).squeeze()

        return logit




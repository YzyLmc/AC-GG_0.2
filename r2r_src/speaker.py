import torch
import numpy as np
from param import args
import os
import utils
import model
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F

class Speaker():
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    def __init__(self, env, listener, tok):
        self.env = env
        self.feature_size = self.env.feature_size
        self.tok = tok
        self.tok.finalize()
        self.listener = listener

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
        if args.pack_lstm:
            self.encoder = model.PackSpeakerEncoder(self.feature_size+4, args.rnn_dim, args.dropout, bidirectional=args.bidir).cuda()
        else:
            self.encoder = model.SpeakerEncoder(self.feature_size+args.angle_feat_size, args.rnn_dim, args.dropout, bidirectional=args.bidir).cuda()
        decoder_class = model.FeedForwardSpeakerDecoder if args.feed_forward else model.SpeakerDecoder
        self.decoder = decoder_class(self.tok.vocab_size(), args.wemb, self.tok.word_to_index['<PAD>'],
                                     args.rnn_dim, args.dropout).cuda()
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def rl_train(self, reward_func, iters, ml_weight=0., policy_weight=1., baseline_weight=.5, entropy_weight=0.,
                 self_critical=False, ml_env=None):
        """
        :param reward_func: A function takes the [(path, inst)] list as input, returns the reward for each inst
        :param iters:       Train how many iters
        :param ml_weight:   weight for maximum likelihood
        :param policy_weight:   weight for policy loss
        :param baseline_weight: weight for critic loss (baseline loss)
        :param entropy_weight:  weight for the entropy
        :param self_critical: Use the self_critical baseline
        :param ml_env:        Specific env for ml (in case that the train_env is aug_env)
        :return:
        """
        from collections import defaultdict
        log_dict = defaultdict(lambda: 0)
        for i in (range(iters)):
            joint_loss = 0.
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Reset Env
            if args.same_in_batch:
                self.env.reset(tile_one=True)
            else:
                self.env.reset()
            rl_batch = self.env.batch

            # RL training
            insts, log_probs, hiddens, entropies = self.infer_batch(sampling=True, train=True) # Sample a batch

            # Get the Reward ( and the length, mask)
            path_ids = [ob['path_id'] for ob in self.env._get_obs()]                # Gather the path ids
            pathXinst = [(path_id, self.tok.shrink(inst)) for path_id, inst in zip(path_ids, insts)]
            reward = reward_func(rl_batch, pathXinst)                               # The reward func will evaluate the instruction
            reward = torch.FloatTensor(reward).cuda()
            length = np.argmax(np.array(insts) == self.tok.word_to_index['<EOS>'], 1) + 1   # Get length (pos of EOS) + 1
            length[length == 1] = insts.shape[1]            # If there is no EOS, change the length to max length.
            mask = 1. - utils.length2mask(length).float()

            # Get the baseline
            if args.normalize_reward:
                baseline = reward.mean()
            else:
                if self_critical:
                    self.env.reset(rl_batch)
                    insts = self.infer_batch(sampling=False, train=False)               # Argmax Decoding
                    pathXinst = [(path_id, self.tok.shrink(inst)) for path_id, inst in zip(path_ids, insts)]
                    baseline = reward_func(rl_batch, pathXinst)                         # The reward func will evaluate the instruction
                    baseline = torch.FloatTensor(baseline).cuda().unsqueeze(1)
                else:
                    baseline_hiddens = hiddens if args.grad_baseline else hiddens.detach()
                    baseline = self.decoder.baseline_projection(baseline_hiddens).squeeze()

            # print("Reward Mean %0.4f, std %0.4f" % (reward.mean().detach().cpu().item(), reward.std().detach().cpu().item()))
            # print("Baseline Mean %0.4f, std %0.4f" % (baseline.mean().detach().cpu().item(), baseline.std().detach().cpu().item()))
            # print("Avg abs(Reward - Baseline): %0.4f" % (torch.abs(reward - baseline).mean().detach().cpu().item()))

            # Calculating the Loss
            reward = reward.unsqueeze(1)            # (batch_size,) --> (batch_size, 1)

            if args.normalize_reward:               # Normalize the reward to mean 0, std 1
                advantage = (reward - baseline) / reward.std() * 0.2
            else:
                advantage = reward - baseline

            policy_loss = (advantage.detach() * (-log_probs) * mask).sum() / self.env.batch_size    # Normalized by the batch_size
            baseline_loss = (advantage ** 2 * mask).sum() / self.env.batch_size
            avg_entropy = (entropies * mask).sum() / self.env.batch_size

            # Add the Loss to the joint_loss
            if baseline_weight != 0.:      # To support the pretrain phase
                joint_loss += baseline_loss * baseline_weight

            if policy_weight != 0.:        # To support the finetune phase
                joint_loss += policy_loss * policy_weight

            if entropy_weight != 0.:        # Note that the negative entrop is added to encourage exploration
                joint_loss += - avg_entropy * entropy_weight

            # ML env preparation
            if ml_env is not None:      # Get the env from ml_env
                old_env = self.env
                self.env = ml_env
                self.env.reset()
            else:                       # else reset the same env as RL
                self.env.reset(batch=rl_batch)

            # ML Training
            assert ml_weight != 0           # Because I always log the ml_weight. And it should always exists!
            if ml_weight != 0.:
                ml_loss = self.teacher_forcing(train=True)
                joint_loss += ml_loss * ml_weight
            else:
                ml_loss = 0.

            if ml_env is not None:
                self.env = old_env

            # print("Reward Mean %0.4f, std %0.4f" % (reward.mean().detach().cpu().item(), reward.std().detach().cpu().item()))
            # print("Baseline Mean %0.4f, std %0.4f" % (baseline.mean().detach().cpu().item(), baseline.std().detach().cpu().item()))
            # print("Avg abs(Reward - Baseline): %0.4f" % (torch.abs(reward - baseline).mean().detach().cpu().item()))

            # Log:
            for name, loss in (('baseline_loss', baseline_loss.detach().item()),
                               ('policy_loss', policy_loss.detach().item()),
                               ('ml_loss', ml_loss.detach().item()),
                               ('baseline', baseline.mean().detach().item()),
                               ('reward', reward.mean().detach().item()),
                               ('baseline_std', baseline.std().detach().item()),
                               ('reward_std', reward.std().detach().item()),
                               ('reward_diff', torch.abs(reward - baseline).mean().detach().item()),
                               ('entropy', avg_entropy.item())):
                log_dict[name] += loss

            # Backward
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        return log_dict

    def get_insts(self, beam=False, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            if beam:
                seqs_list = self.beam_infer_batch()
                # If existing a beam, return the most possible one
                insts = [[] if len(seqs) == 0 else seqs[0] for seqs in seqs_list]
            else:
                insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, iters=None, beam=False, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(beam, *aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if args.fast_train else 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        first_feat = np.zeros((len(obs), self.feature_size+args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            assert not args.pack_lstm and insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            if args.pack_lstm:
                (img_feats, can_feats, first_feat), lengths = self.from_shortest_path(get_first_feat=True)      # Image Feature (from the shortest path)
                ctx = self.encoder(can_feats, img_feats, lengths, first_feat)
            else:
                (img_feats, can_feats), lengths = self.from_shortest_path()      # Image Feature (from the shortest path)
                ctx = self.encoder(can_feats, img_feats, lengths)
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature

        # Decode
        logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)
        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if for_listener:
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            )

        if train:
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        if args.pack_lstm:
            (img_feats, can_feats, first_feat), lengths = self.from_shortest_path(get_first_feat=True)      # Image Feature (from the shortest path)
        else:
            (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            # Note: first feat doesn't need dropout
            if args.dropviewpoint:
                # assert len(viewpoints_list[0]) == img_feats.size(1)
                # assert len(viewpoints_list) == img_feats.size(0) == batch_size
                for i, viewpoints in enumerate(viewpoints_list):
                    for j, viewpoint in enumerate(viewpoints):
                        img_feats[i, j, ..., :-args.angle_feat_size] *= featdropmask[i][viewpoint]     # batch_id --> (viewpoint --> mask)
                        # can_feats[i, j, :-args.angle_feat_size] *= featdropmask[i][viewpoint]     # batch_id --> (viewpoint --> mask)
            else:
                img_feats[..., :-args.angle_feat_size] *= featdropmask.view(-1, 1, 1, self.feature_size)
                can_feats[..., :-args.angle_feat_size] *= featdropmask.view(-1, 1, self.feature_size)

        # Encoder
        if args.pack_lstm:
            ctx = self.encoder(can_feats, img_feats, lengths, first_feat,
                               already_dropfeat=(featdropmask is not None))
        else:
            ctx = self.encoder(can_feats, img_feats, lengths,
                               already_dropfeat=(featdropmask is not None))
        ctx_mask = utils.length2mask(lengths)

        # Decoder
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.bool)
        word = np.ones(len(obs), np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1)
        else:
            return np.stack(words, 1)       # [(b), (b), (b), ...] --> [b, l]

    def beam_infer_batch(self, beam_size=5, seq_num=20, candidates=20):
        """

        :param beam_size:  Beam_size is the size of the beam-search
        :param seq_num:    Seq_num is the maximum number of returned sequence
        :param candidates: The maximum number of candidate sequences
        :return: [[seq 1, seq 2, ... (seq_num in total)] (for batch 1),
                  [seq 1, seq 2, ... (seq_num in total)] (for batch 2),
                  ...,
                  [seq 1, seq 2, ... (seq_num in total)] (for batch n)]
        """
        # Eval Model
        self.encoder.eval()
        self.decoder.eval()

        # Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        (img_feats, can_feats), lengths = self.from_shortest_path()  # Feature from the shortest path

        # Encoder
        ctx = self.encoder(can_feats, img_feats, lengths)  # Encode
        ctx_mask = utils.length2mask(lengths)

        # init of the Deocer
        results = []
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.int)
        word = np.ones(len(obs), np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()

        # Beam Search Initialization
        bs_now = 1
        pre_scores = torch.zeros((batch_size, bs_now))
        vocab_size = self.tok.vocab_size()
        for t in range(args.maxDecode):
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)  # Decode, logits: (b, 1, vocab_size)
            logits = logits.view(batch_size, bs_now, -1)                    # logits: (b, beam_size, vocab_size)

            log_prob = F.log_softmax(logits, dim=2).cpu()                   # logit --> log_softmax--> log_prob
            scores = pre_scores.unsqueeze(-1) + log_prob                    # scores: (batch, beam, vocab_size)

            # select top beam_size words. save it
            scores, word = scores.view(batch_size, -1).topk(beam_size, dim=1)
            beam = word / vocab_size                                    # beam: (batch, beam) [[0,1,1], [0,1,2]
            word = word % vocab_size

            # Log the result
            for i in range(batch_size):
                if ended[i] >= candidates:          # if the maximum seq exceeded, don't add it
                    word[i] = self.tok.word_to_index['<PAD>']
            results.append({"beam": beam, "word": word, "scores": scores.detach().clone()})  # Save it before change the scores

            # For next step
            beam = beam + torch.arange(batch_size, dtype=torch.int64).view(-1, 1) * bs_now  #  [[0,1,1], [3,4,5], ..
            def gather_beam(state):                                      # State: (batch * beam, rnn_dim)
                return state[:, beam.view(-1)]
            h_t, c_t = (gather_beam(state) for state in (h_t, c_t))
            pre_scores = scores
            bs_now = beam_size
            assert bs_now == beam.size(1)

            # Handle the end_beams by setting the pre_scores to a very small value
            for i in range(word.size(0)):
                flag = True
                for j in range(word.size(1)):
                    if word[i][j] == self.tok.word_to_index['<EOS>']:
                        pre_scores[i][j] = -float('inf')        # Set the score to -inf (so it will not appear in next step)
                        ended[i] += 1                           # One more <end> seq for batch i
                    else:
                        flag = False
                if flag:            # If all ended, set it to maximum
                    ended[i] = candidates
                #assert not flag         # If all the beams want to end, just stop here.

            # At last, change the input
            word = word.view(-1, 1).cuda()

            # Should it stop now?
            if (ended >= candidates).all():
                break

        seqs = self.get_all_ends(results, batch_size)
        results = []
        for i in range(batch_size):
            # sorted_seq = sorted(seqs[i], key=lambda x: x['score'] / len(x['inst']), reverse=True)
            # sorted_seq = sorted(seqs[i], key=lambda x: x['score'] - 0.5 * abs(29 - len(x['inst'])), reverse=True)
            sorted_seq = sorted(seqs[i], key=lambda x: x['score'], reverse=True)
            # print(sorted_seq)
            results.append([list(seq['inst']) for seq in sorted_seq[:seq_num]])

        # print()

        # for seq in results[0]:
        #     print(self.tok.decode_sentence(seq))

        return results      # [[inst_1, inst_2, ..., inst_{seq_num}], ... ]

    def get_all_ends(self, results, batch_size):
        from collections import deque
        seqs = [list() for _ in range(batch_size)]
        for state in reversed(results):
            beam = state['beam'].numpy()        # The beam in the previous decoding state
            word = state['word'].numpy()        # The word in beam-search
            scores = state['scores'].numpy()    # The sum_log_probs score
            for i in range(batch_size):
                for seq in seqs[i]:
                    seq['inst'].appendleft(word[i][seq['last_beam']])
                    seq['last_beam'] = beam[i][seq['last_beam']]
                for w, b, s in zip(word[i], beam[i], scores[i]):
                    if w == self.tok.word_to_index['<EOS>']:    # If the word is <EOS>, retracking
                        seqs[i].append({
                            "inst": deque([w]),
                            "last_beam": b,
                            "score": s
                        })
        return seqs

    def search_inst(self):
        self.env.reset_epoch(shuffle=True)
        total = self.env.size()
        for _ in tqdm(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            batch = self.env.batch.copy()
            insts = self.infer_batch(train=False)  # Get the insts of the result
            arg_inst = self.tok.decode_sentence(self.tok.shrink(insts[1]))
            print(arg_inst)
            for feat_id in range(2048):
                self.env.reset(batch)
                self.listener.decoder.train()
                # drop_mask = self.listener.decoder.drop3(torch.ones(2048).cuda())
                drop_mask = torch.ones(2048).cuda()
                drop_mask[feat_id] = 0.
                insts = self.infer_batch(featdropmask=drop_mask)  # Get the insts of the result
                # insts = self.infer_batch(train=True, featdropmask=None)  # Get the insts of the result
                # path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
                for inst in insts[1:2]:
                    sent = (self.tok.decode_sentence(self.tok.shrink(inst)))
                    if "fireplace" not in sent:
                        print(feat_id, sent)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1


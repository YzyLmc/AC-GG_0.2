#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:02:29 2021

@author: ziyi
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import try_cuda

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):
    '''Like SoftDot, but don't concatenat h or perform the non-linearity transform
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn
    
class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        visual_context: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)  # batch x v_dim
        return weighted_context, attn
    
    
class geEncoder(nn.Module):
    def __init__(self, action_size, world_size, hidden_size, dropout_ratio):
        super(geEncoder,self).__init__()
        
        self.action_size = action_size
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.attention_layer = VisualSoftDotAttention(hidden_size,world_size)
        self.lstm = nn.LSTMCell(action_size + world_size,hidden_size)
        self.encoder2decoder = nn.Linear(hidden_size,hidden_size)
    def init_state(self, batch_size):
        h0 = Variable(torch.zeros(batch_size,self.hidden_size),requires_grad = False)
        c0 = Variable(torch.zeros(batch_size,self.hidden_size),requires_grad = False)
        
        return try_cuda(h0), try_cuda(c0)
    
    def forward(self, batched_action, batched_world_state):
        assert len(batched_world_state) == len(batched_action)
        batch_size = len(batched_action)
        
        h,c = self.init_state(batch_size)
        h_ls = []
        
        for t, (action, world_state) in enumerate(zip(batched_action,batched_world_state)):
            weighted_feature = self.visual_layer(h, world_state)
            concat_af = torch.cat((action,weighted_feature))
            dropped = self.drop(concat_af)
            h, c = self.lstm(dropped, (h, c))
            h_ls.append(h)
            
        decoder_init = nn.Tanh()(self.encoder2decoder(h))
        
        #ctx = self.drop(ctx)
        return ctx, decoder_init, c
    
    
class geDecoder(nn.Module):
    def __init__(self, vocab_size,vocab_embedding_size, hidden_size, dropout_ratio, glove = None, use_input_att_feed= False):
        super(geDecoder, self).__init__()
        self.vocab_size =   vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        
        self.drop = nn.Dropout(p = dropout_ratio)
        self.use_input_att_feed = use_input_att_feed
        if self.use_input_att_feed:
            print('using attention feed')
            self.lstm = nn.LSTMCell(vocab_embedding + hidden_size, hidden_size)
            self.output_l1 = nn.Linear(hidden_size*2,hidden_size)
            self.tanh = nn,Tanh()
            
        else:
            self.lstm = nn.LSTMCell(vocab_embedding,hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)
            
        self.decoder2word = nn.Linear(hidden_size, vocab_embedding_size)
        
    def foward(self, p_word, h0, c0, ctx, ctx_mask = None):
        word_embeds  = self.embedding(p_word)
        word_embeds = word_embeds.squeeze() # (batch,emedding_size)
        if not self.use_glove:
            word_embeds_dropped = self.drop(word_embeds)
        else:
            word_embeds_dropped = word_embeds
            
        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_dropped, self.drop(h_tilde)),1)
            h_1, c_1 =self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2action(x)
            
        else:
            h_1, c_1 = self.lstm(word_embeds_dropped, (h_0, c_0))
            h_1_drop = self.drop(h_1)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit = self.decoder2word(h_tilde)
        
        return h_1, c_1, alpha, logit
    
class 
            
            
            
            
            
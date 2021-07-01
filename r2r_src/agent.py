''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer, read_img_features
import utils
import model
import env
import param
from param import args
from collections import defaultdict

#from allennlp.commands.elmo import ElmoEmbedder

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {} 
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                if i > 0:
                    kwargs['show'] = False
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    kwargs['show'] = False
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
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

    def __init__(self, env, results_path, tok, feat=None, candidates=None, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        if env:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 2048
        # Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        self.encoder = model.EncoderLSTM(tok.vocab_size(), args.wemb, enc_hidden_size, padding_idx,
                                         args.dropout, bidirectional=args.bidir).cuda()
        if args.elmo:
            self.ee = ElmoEmbedder(cuda_device=0)
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.rnn_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.critic = model.Critic().cuda()
        self.point_angle_feature = utils.get_point_angle_feature()
        self.models = (self.encoder, self.decoder, self.critic)
        # print(self.point_angle_feature)

        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

        # Loss for the denoise
        self.criterion_denoise = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='none')
        self.gate_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        # Logs
        sys.stdout.flush()
        self.timer = utils.Timer()
        self.logs = defaultdict(list)
        
        # for ground function-Ziyi
        self.buffered_state_dict = {}
        self.sim = utils.new_simulator()
        self.angle_feature = utils.get_all_point_angle_feature()
        if feat:
            self.features = feat
        if candidates:    
            self.candidates = candidates


    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)

        if "detectfeat" in args.features:   # The detection feature will contain the angle feat in it!
            for i, ob in enumerate(obs):
                features[i, :, :] = ob['feature']       # Image + position feat
        else:
            for i, ob in enumerate(obs):
                features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate at len(candidate) is the feature for the END
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        # if args.features == 'semimg':
        #     candidate_feat[..., :self.feature_size] -= 0.5
        #     f_t[..., :self.feature_size] -= 0.5

        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
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
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    # print("right")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self, show=False, train_ml=None, train_rl=True, reset=True, denoise=False,
                make_noise=False, speaker=None):
        """

        :param show:
        :param train_ml:
        :param train_rl:
        :param reset:
        :param make_noise:  None --> Do not make noise, False --> Make default noise in listener
                            True --> Explicit make noise
        :param speaker:
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        self.timer.tic("prepare")

        if reset:
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        noise = None
        if make_noise and self.decoder.training:
            assert (int(args.dropbatch) + int(args.dropinstance) + int(args.dropviewpoint)) <= 1
            if args.dropbatch:
                noise = self.decoder.drop3(torch.ones(self.feature_size).cuda())
            elif args.dropinstance:
                noise = self.decoder.drop3(torch.ones(batch_size, self.feature_size).cuda())
            elif args.dropviewpoint:
                noise = np.array(
                    [defaultdict(lambda: self.decoder.drop3(torch.ones(36, self.feature_size).cuda()))
                     for _ in range(batch_size)]
                )
            if (args.dropinstance or args.dropbatch):   # Something that you don't want to dropout. Because they are critical to the structral of the room
                if args.features == "semlabnorm":
                    noise[..., param.NODROPCAT] = 1. / (1. - args.featdropout)
                    if args.dropmask01:
                        noise[noise > 0.0001] = 1.
                if args.features == 'label1000':
                    if args.topk != -1: # Do not dropout the topk classes in imagenet feature.
                        noise[..., param.LABEL1000_TOPK] = 1. / (1. - args.featdropout)
                if "detectlab" in args.features:
                    """
                    wall 58406614422.394295
                    floor 17664313038.707993
                    room 15723545913.166748
                    ceiling 15177922795.981789
                    building 9737031162.760834
                    door 7045039380.30793
                    window 6903454482.880592
                    """
                    noise[..., :7] = 1. / (1. - args.featdropout)
                    if args.dropmask01:
                        noise[noise > 0.0001] = 1.

        # print(make_noise, speaker, noise, args.dropspeaker)
        if speaker is not None:     # Trigger the self_train mode!
            batch = self.env.batch.copy()
            # print(obs[0]['instructions'])
            # print(batch[0]['instructions'])
            # print(obs[0]['instr_encoding'])
            speaker.env = self.env
            if args.dropspeaker:        # if need drop_speaker
                insts = speaker.infer_batch(train=True)
            else:
                insts = speaker.infer_batch(featdropmask=noise)
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))
            # print(obs[0]['instructions'])
            # print(obs[0]['instr_encoding'])
            # print()

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        if noise is not None:
            if len(noise) == batch_size:
                noise = noise[perm_idx]

        # This part is just for ablation and will not be updated
        if args.independent_drop:   # Otherwise, tying the dropout.
            if make_noise and self.decoder.training:
                if args.dropbatch:
                    noise = self.decoder.drop3(torch.ones(self.feature_size).cuda())
                if args.dropinstance:
                    noise = self.decoder.drop3(torch.ones(batch_size, self.feature_size).cuda())
                if args.dropviewpoint:
                    noise = np.array(
                        [defaultdict(lambda: self.decoder.drop3(torch.ones(self.feature_size).cuda()))
                         for _ in range(batch_size)]
                    )

        # ELMO embedding extraction
        # if args.elmo:
        #     insts = [Tokenizer.split_sentence(ob['instructions'])[:79] for ob in perm_obs]
        #     embed, elmo_mask = self.ee.batch_to_embeddings(insts)
        #     elmo_mask = 1 - elmo_mask     # In the original program, the mask is to set float('-inf')
        #     elmo_mask = elmo_mask.byte()
        #     seq_lengths = [l - 1 for l in seq_lengths]  # Do not need EOS for now
        #     for i in range(batch_size):
        #         assert seq_lengths[i] == int((1 - elmo_mask).sum(1)[i])
        #
        #     # Forward through encoder, giving initial hidden state and memory cell for decoder
        #     ctx, h_t, c_t = self.encoder(embed, seq_lengths)
        #     ctx_mask = elmo_mask
        # else:
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        if args.fixed_size_ctx > 0:
            ctx_mask = None
        else:
            ctx_mask = seq_mask

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For debugging
        start_distance = [ob['distance'] for ob in perm_obs]
        views_list = [list() for _ in perm_obs]
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        ml_losses = []

        # Initialize the attention in the loop.
        length = ctx.size(1)
        alpha = torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, length - 1)], 1)
        alpha = Variable(alpha, requires_grad=False).cuda()

        prev_feat = Variable(torch.zeros(batch_size, args.views, self.feature_size+args.angle_feat_size), requires_grad=False).cuda()
        h1 = h_t
        self.timer.toc("prepare")
        for t in range(self.episode_len):

            self.timer.tic("feat")
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if noise is not None:
                if args.dropviewpoint:  # Dropout the viewpoint by the mask saved in the defaultdict
                    for i, ob in enumerate(perm_obs):
                        # if ob['viewpoint'] not in noise[i]:
                        #     print(i, "not in noise")
                        assert not train_ml or ob['viewpoint'] in noise[i]
                        # candidate_feat[i, ..., :-args.angle_feat_size] *= noise[i][ob['viewpoint']]
                        f_t[i, ..., :-args.angle_feat_size] *= noise[i][ob['viewpoint']]
                else:
                    candidate_feat[..., :-args.angle_feat_size] *= noise.view(-1, 1, self.feature_size)
                    f_t[..., :-args.angle_feat_size] *= noise.view(-1, 1, self.feature_size)

            self.timer.toc("feat")

            self.timer.tic("decoder")
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat, candidate_leng,
                                                      prev_feat, h_t, h1, c_t, alpha, ctx, ctx_mask,
                                                      noise is not None or make_noise is None)
            self.timer.toc("decoder")

            self.timer.tic("states")
            prev_feat = f_t
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)
            self.timer.toc("states")

            self.timer.tic("sample")
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            self.timer.toc("sample")

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            self.timer.tic("action")
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1
            self.timer.toc("action")

            # Make action and get the new state
            self.timer.tic("step")
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu
            self.timer.toc("step")

            self.timer.tic("reward")
            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            raise NameError("The action doesn't change the move")
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist
            self.timer.toc("reward")

            # For debugging
            for i, ob in enumerate(perm_obs):
                views_list[i].append((ob['viewpoint'][-5:], "act %d/%d" % (cpu_a_t[i], candidate_leng[i]-1),
                                      "%0.4f"%dist[i]))

            self.timer.tic("track")

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break
            self.timer.toc("track")

        # import pprint
        # if self.feedback == 'argmax':       # In Test
        #     value_list = []
        #     for hidden in hidden_states:
        #         value_list.append(self.critic(hidden).detach())
        #     for i in range(batch_size):
        #         print("Env %s" % self.env.name)
        #         print("Inst Id %s" % perm_obs[i]['instr_id'])
        #         print("Scan %s" % perm_obs[i]['scan'])
        #         # pprint.pprint(views_list[i])
        #         for j, view_info in enumerate(views_list[i]):
        #             value = value_list[j][i].item()
        #             print(j, view_info, "%0.4f" % value)
        #         print(perm_obs[i]['instructions'])
        #         print("Start Distance is %0.4f" % start_distance[i])
        #         print('Final Distance is %0.4f' % perm_obs[i]['distance'])
        #         print()

        # for i in range(batch_size):
        #     print("Inst Id %s" % obs[i]['instr_id'])

        # if not self.decoder.training and show:

        #     print()

        self.timer.tic("a2c")
        if train_rl:
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if noise is not None:
                if args.dropviewpoint:  # Dropout the viewpoint by the mask saved in the defaultdict
                    for i, ob in enumerate(perm_obs):
                        # candidate_feat[i, ..., :-args.angle_feat_size] *= noise[i][ob['viewpoint']]
                        f_t[i, ..., :-args.angle_feat_size] *= noise[i][ob['viewpoint']]
                else:
                    candidate_feat[..., :-args.angle_feat_size] *= noise.view(-1, 1, self.feature_size)
                    f_t[..., :-args.angle_feat_size] *= noise.view(-1, 1, self.feature_size)
            last_h_, _, _, _, _ = self.decoder(input_a_t, f_t, candidate_feat, candidate_leng,
                                               prev_feat, h_t, h1, c_t, alpha, ctx, ctx_mask,
                                               noise is not None or make_noise is None)
            rl_loss = 0.
            # hidden_states, policy_log_probs, rewards
            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            # print("RL_loss", rl_loss)
            self.loss += rl_loss

        if train_ml is not None:
            # print("ML_LOSS", ml_loss / batch_size, "with_weight", train_ml)
            self.loss += ml_loss * train_ml / batch_size

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        self.timer.toc("a2c")
        self.timer.step()

        return traj
    
    def make_candidate(self, feature, candidate, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        id2feat = {c['viewpointId']: c['feature'] for c in candidate}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        #if True:
                            #visual_feat = id2feat[loc.viewpointId]
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                 'pointId', 'idx']}
            for c in candidate]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new
        
    def make_action(self,sim, a_t, obs, traj=None):
        def take_action(name):
            if type(name) is int:       # Go to the next view
                sim.makeAction(name, 0, 0)
            else:                       # Adjust
                sim.makeAction(*self.env_actions[name])
            state = sim.getState()
            if traj is not None:
                traj[0]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                if type(name) is int:
                    traj[0]['traj'].append(state.location.viewpointId)

        action = a_t[0]
        if action != -1:            # -1 is the <stop> action
            select_candidate = obs[0]['candidate'][action]
            src_point = obs[0]['viewIndex']
            trg_point = select_candidate['pointId']
            src_level = (src_point ) // 12   # The point idx started from 0
            trg_level = (trg_point ) // 12
            while src_level < trg_level:    # Tune up
                take_action('up')
                src_level += 1
            while src_level > trg_level:    # Tune down
                take_action('down')
                src_level -= 1
            while sim.getState().viewIndex != trg_point:    # Turn right until the target
                # print("right")
                # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                take_action('right')
            
            take_action(select_candidate['idx'])
    
    def observe(self, sim):
        obs = []
        #for i, (feature, candidate, state) in enumerate(self.env.getStates()):
        # print("Get obs %d"%i)
        # sys.stdout.flush()
        state = sim.getState()
        base_view_id = state.viewIndex
        long_id = state.scanId + '_' + state.location.viewpointId
        if self.features:
            feature = self.features[long_id]     # Get feature for
            candidate = self.candidates[long_id]
        # Full features
        candidate = self.make_candidate(feature, candidate, state.scanId, state.location.viewpointId, state.viewIndex)

        # By using this, only the heading is shifted, the angle_feature is added.
        # candidate = self.make_simple_candidate(candidate, base_view_id)

        # (visual_feature, angel_feature) for views
        feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
        obs.append({
            
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : feature,
            'candidate': candidate,
            'navigableLocations' : state.navigableLocations           
            
            
        })
        #if 'instr_encoding' in item:
        #    obs[-1]['instr_encoding'] = item['instr_encoding']
        # A2C reward. The negative distance between the state and the final state
        #obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs
    
    def ground(self, sim, encoded_instructions, scanId, viewpointId,heading = 0.0, elevation = 0.0):
                        
        sim.newEpisode(scanId, viewpointId,heading, elevation)        
        
        initial_obs = self.observe(sim)
        batch_size = len(initial_obs)
        
        self.encoder.eval()
        self.decoder.eval()
        
        seq = encoded_instructions.unsqueeze(0)
        seq_lengths = [len(encoded_instructions)]
        
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)    #encode instr
        
        ctx_mask = None
        
                # Record starting point
        traj = [{
            'traj': [ob['viewpoint']],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in initial_obs]

        
        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        ml_losses = []
        # Initialization the tracking state
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env
        
        # Initialize the attention in the loop.
        length = ctx.size(1)
        alpha = torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, length - 1)], 1)
        alpha = Variable(alpha, requires_grad=False).cuda()
        
        prev_feat = Variable(torch.zeros(batch_size, args.views, self.feature_size+args.angle_feat_size), requires_grad=False).cuda()
        h1 = h_t
        
        obs = initial_obs
        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)
            
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat, candidate_leng,
                                                      prev_feat, h_t, h1, c_t, alpha, ctx, ctx_mask
                                                      #noise is not None or make_noise is None
                                                      )

            prev_feat = f_t
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)

            logit.masked_fill_(candidate_mask, -float('inf'))


            _, a_t = logit.max(1)        # student forcing - argmax
            a_t = a_t.detach()
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch


            # Prepare environment action
            # NOTE: Env action is in the perm_obs space

            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_action(sim, cpu_a_t, obs, traj= traj)
            obs = self.observe(sim)                         

            

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break
            


        return traj

            
            
    def _beam_search(self):
        """
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))
                # print(viewpoint, results[i]['dijk_path'])

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat, candidate_leng,
                                                      None, h_t, h1, c_t, None, ctx, ctx_mask, False)

            # Update the states with the new viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    # new_score = current_state['score'] + log_probs[i][j].detach().cpu().item() #/ (len(candidate) + 1)
                    modified_log_prob = log_probs[i][j].detach().cpu().item() #/ (len(candidate) + 1)
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            ]
        }
        """
        self.env.reset()
        results = self._beam_search()
        """
        return from self._beam_search()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results


    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters, show=True)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, denoise=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, denoise=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, denoise=False, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()


    def train(self, n_iters, feedback='teacher', denoise=False, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            # try:
            if True:
                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.teacher_weight, train_rl=False, denoise=False, **kwargs)
                elif feedback == 'sample':
                    # if 'reset' in kwargs and not kwargs['reset']:   # reset the batch or not
                    #     pass
                    # else:
                    #     self.env.reset()
                    # batch = self.env.batch                          # Save the current batch
                    if args.ml_weight != 0:
                        self.feedback = 'teacher'
                        self.rollout(train_ml=args.ml_weight, train_rl=False, denoise=False, **kwargs)
                    # self.env.reset(batch)                           # Reset to the batch used in IL (for RL)
                    self.feedback = 'sample'
                    self.rollout(train_ml=None, train_rl=True, denoise=False, **kwargs)
                else:
                    assert False

            # except Exception as inst:
            #     print("ERROR OCCUR")
            #     print(type(inst))
            #     print(inst.args)
            #     print(inst)
            #     self.loss = 0
            #     continue

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()

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
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print("Recovering the model for %s" % (name))
            # print(list(model.state_dict().keys()))
            # model.load_state_dict(states[name]['state_dict'])
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            #if (len(model_keys), len(load_keys))
            # # print(model_keys.difference(load_keys))
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                # print("Loading the Optimizer parameter")
                optimizer.load_state_dict(states[name]['optimizer'])
            # else:
            #     print("NOT Loading the Optimizer parameter")
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1


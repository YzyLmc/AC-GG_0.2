
import torch
import sys
import numpy as np
from param import args
import os
import utils
import model
import torch.nn.functional as F

class Arbiter():
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
        print("Start Arbiter Init")
        self.env = env
        self.tok = tok
        self.listener = listener

        # Model
        # self.bidaf = model.BidirectionalArbiter(tok.vocab_size(), 2048 + 4, args.wemb,
        #                                         self.tok.word_to_index['<PAD>'],
        #                                         args.rnn_dim, args.dropout).cuda(0)
        # self.bidaf_optimizer = args.optimizer(self.bidaf.parameters(), lr=args.lr)
        self.bidaf = model.BidirectionalArbiter(tok.vocab_size(), 2048 + 4, args.wemb,
                                                self.tok.word_to_index['<PAD>'],
                                                args.rnn_dim, 0.0).cuda(0)
        self.bidaf_optimizer = torch.optim.Adam(self.bidaf.parameters(), lr=1e-4)
        self.iter = 0

        # Evaluation
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def train(self, iters):
        for i in range(iters):
            self.bidaf_optimizer.zero_grad()

            self.env.reset()
            loss = self.teacher_forcing()

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.bidaf.parameters(), 40.)
            self.bidaf_optimizer.step()

    def get_probs(self, total=None, wrapper=(lambda x: x)):
        # Get the caption for all the data
        path2prob = {}
        if total is None:   # Evaluate the whole dataset
            total = self.env.size()
            self.env.reset_epoch(shuffle=False)
        else:               # Evaluate a random sample of total data
            self.env.reset_epoch(shuffle=True)
        iters = total // self.env.batch_size + 1
        for _ in wrapper(range(iters)):     # Guarantee that all the data are processed
            obs = self.env.reset()
            probs = self.infer_batch()      # Get the probs of the result
            inst_ids = [ob['instr_id'] for ob in obs]  # Gather the inst ids
            for path_id, prob in zip(inst_ids, probs):
                if path_id not in path2prob:
                    path2prob[path_id] = float(prob)
        return path2prob

    def valid(self, *args, **kwargs):
        """
        """
        arbiter_env = self.env

        self.env = arbiter_env.gt
        inst2prob = self.get_probs(*args, **kwargs)
        self.env = arbiter_env.gen
        inst2prob.update(self.get_probs(*args, **kwargs))

        self.env = arbiter_env

        return inst2prob

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
                src_level = (src_point + 1) // 12   # The point idx started from 0
                trg_level = (trg_point + 1) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
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
        candidate_feat = np.zeros((len(obs), 2048 + 4), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :args.feature_size] = c['feature'] # Image feat
                candidate_feat[i, -4:] = utils.angle_feature(c['heading'], c['elevation'])   # Position Feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self):
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        while not ended.all():
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
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len,
        return (img_feats, can_feats), length

    def gt_words(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_len = (seq_tensor != self.tok.word_to_index['<PAD>']).sum(1)    # The number of word that is not PAD
        return torch.from_numpy(seq_tensor).cuda(), seq_len

    def train_one_step(self, env, batch, insts):
        self.train_gt(env)
        self.train_fake(batch, insts)

    def train_gt(self, env, label=1.):
        """
        NOTE: It does not reset the ENV!!!!!!!!!
        :param env: The env, need to be reset
        :param label: The label for the data
        :return:
        """
        self.bidaf_optimizer.zero_grad()

        self.env = env
        loss = self.teacher_forcing(train=True, target=label)

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.bidaf.parameters(), 40.)
        self.bidaf_optimizer.step()

        return loss.item()

    def train_fake(self, batch, insts):
        self.bidaf_optimizer.zero_grad()

        loss = self.infer_batch(batch=batch, insts=insts, train=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.bidaf.parameters(), 40.)
        self.bidaf_optimizer.step()

        return loss.item()

    def teacher_forcing(self, train=True, target=None):
        if train:
            self.bidaf.train()
        else:
            self.bidaf.eval()

        # Get input
        obs = self.env._get_obs()
        (img_feats, can_feats), feat_len = self.from_shortest_path()      # Feature from the shortest path
        insts, inst_len = self.gt_words(obs)

        # Get Ground Truth Label
        if target is None:      # Label from the env
            target = np.array([ob['label'] for ob in obs], np.float32)
            target = torch.from_numpy(target).cuda()
        else:
            target = torch.FloatTensor([target] * self.env.batch_size).cuda()

        feat_mask = utils.length2mask(feat_len)
        inst_mask = utils.length2mask(inst_len, 80)
        logits = self.bidaf(img_feats, can_feats, feat_mask, insts, inst_mask)
        # print("TRUE:", torch.sigmoid(logits).mean())

        loss = self.bce_loss(input=logits, target=target)

        if train:
            return loss
        else:
            return loss.item()

    def infer_batch(self, batch=None, insts=None, train=False):
        """
        :param insts:  numpy array with [batch_size, length]. It should be PADDED
        :return: The prob numpy with [batch_size]
        """
        if train:
            self.bidaf.train()
        else:
            self.bidaf.eval()

        # Get Visual Input
        if batch is not None:
            self.env.reset(batch)
        obs = self.env._get_obs()
        (img_feats, can_feats), feat_len = self.from_shortest_path()      # Feature from the shortest path

        # Get Language Input
        if insts is None:
            # Use the default inst in the dataset if the argument **insts** is not given
            insts, inst_len = self.gt_words(obs)
        else:
            # Bring the numpy to cuda
            # Use FloatTensor() so insts could be another Tensor
            if type(insts) is list:
                max_length = max([len(inst) for inst in insts])
                insts = [inst + ([self.tok.word_to_index['<PAD>']] * (max_length - len(inst)))
                         for inst in insts]
                insts = np.array(insts)

            # print("G infer", self.tok.decode_sentence(insts[0]))
            inst_len = (insts != self.tok.word_to_index['<PAD>']).sum(1)
            # print("len", inst_len[0])
            insts = torch.LongTensor(insts).cuda()

        # Create Mask
        feat_mask = utils.length2mask(feat_len)
        inst_mask = utils.length2mask(inst_len, insts.size(1))

        # input --> logit --> probs --> cpu_probs
        logits = self.bidaf(img_feats, can_feats, feat_mask, insts, inst_mask)
        # print("FALSE:", torch.sigmoid(logits).mean())

        if train:
            target = torch.FloatTensor([0.] * self.env.batch_size).cuda()
            loss = self.bce_loss(input=logits, target=target)
            return loss
        else:
            probs = torch.sigmoid(logits)
            answer = probs.cpu().detach().numpy()
            return answer

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
        all_tuple = [("model", self.bidaf, self.bidaf_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            print(name)
            print(list(model.state_dict().keys()))
            model.load_state_dict(states[name]['state_dict'])
            optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("model", self.bidaf, self.bidaf_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['model']['epoch'] - 1


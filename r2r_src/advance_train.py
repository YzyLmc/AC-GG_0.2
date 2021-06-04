
import torch

import os
import time
import json
import random
import sys
import numpy as np
from collections import defaultdict
from speaker import Speaker
from arbiter import Arbiter

import tqdm

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, SemiBatch, ArbiterBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import IPython

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
IMAGENET_CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'

PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
PLACE365_CANDIDATE_FEATURES = 'img_features/ResNet-152-places365-candidate.tsv'

if args.place365 or args.features == 'place365':
    print("USE FEATURE PLACE365")
    features = PLACE365_FEATURES
    CANDIDATE_FEATURES = PLACE365_CANDIDATE_FEATURES
elif args.features == 'semlab':
    features = 'img_features/Sem-lab-view.tsv'
    CANDIDATE_FEATURES = 'img_features/Sem-lab-candidate.tsv'
elif args.features == 'semlabnorm':
    features = 'img_features/Sem-lab-normalize-view.tsv'
    CANDIDATE_FEATURES = 'img_features/Sem-lab-normalize-candidate.tsv'
elif args.features == 'semlabpos':
    features = 'img_features/Sem-lab-pos-view.tsv'
    CANDIDATE_FEATURES = 'img_features/Sem-lab-pos-candidate.tsv'
elif args.features == 'semimg':
    features = 'img_features/Sem-img-view.tsv'
    CANDIDATE_FEATURES = 'img_features/Sem-img-candidate.tsv'
elif args.features == 'imagenet':
    features = IMAGENET_FEATURES
    CANDIDATE_FEATURES = IMAGENET_CANDIDATE_FEATURES
elif args.features == 'label1000':
    features = "img_features/ResNet-152-Label-views.tsv"
    CANDIDATE_FEATURES = "img_features/ResNet-152-Label-candidate.tsv"
elif args.features == 'label50':
    features = "img_features/ResNet-152-Label50-views.tsv"
    CANDIDATE_FEATURES = "img_features/ResNet-152-Label50-candidate.tsv"
elif "detectlab" in args.features:
    num = int(args.features[9:])
    print("Use the detection feature %d" % num)
    features = "img_features/detection-lab%d-view.tsv" % (num)
    CANDIDATE_FEATURES = "img_features/detection-lab%d-candidate.tsv" % (num)
elif "detectfeat" in args.features:
    num = int(args.features[10:])
    print("Use the detection feature with %d objs" % num)
    features = "img_features/detection-feat%d-view.tsv" % (num)
    CANDIDATE_FEATURES = IMAGENET_CANDIDATE_FEATURES
elif args.features == 'pca':
    features = 'img_features/ResNet-152-PCA300.tsv'
    CANDIDATE_FEATURES = IMAGENET_CANDIDATE_FEATURES
else:
    assert False, "The feature %s is not provieded" % args.features


if args.fast_train:
    name, ext = os.path.splitext(features)

    features = name + "-fast" + ext


print("Use view features %s and candidate feature %s" % (features, CANDIDATE_FEATURES))


# features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
feedback_method = args.feedback # teacher or sample


print(args)


def train_rl_speaker(train_env, tok, n_iters, log_every=100, val_envs={}):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    major_metric = args.metric             # The metric for reward and for early stopping

    reward_func = lambda batch, pathXinst: val_envs['train'][1].batch_lang_score(pathXinst, metric=major_metric, reduce=False)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE DICT from %s" % args.load)
        tmp = speaker.load(args.load)
        if "rl" in args.load:           # If load from current run
            start_iter = tmp

    if args.fast_train:
        log_every = 50

    pretrain_iters = -100 if args.self_critical else 200     # No pretrain for self_critical
    assert pretrain_iters % log_every == 0
    best_score = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 9595)
    for idx in range(start_iter, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        idx_tobe = idx + interval

        print("Iter: %05d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()     # The dict here is to avoid multiple evaluation for one path
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[path_id]['instructions'])
            avg_length = utils.average_length(path2inst)
            pathXinst = list(path2inst.items())
            name2score = evaluator.lang_eval(pathXinst, no_metrics=('METEOR',))
            major_score = name2score[major_metric]
            score_string = " "
            for score_name, score in name2score.items():
                writer.add_scalar("lang_score/%s/%s" % (score_name, env_name), score, idx)
                score_string += "%s_%s: %0.4f " % (env_name, score_name, score)

            # Tensorboard log
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("avg_length/%s" % (env_name), avg_length, idx)

            # Save the model according to the bleu score
            if major_score > best_score[env_name]:
                best_score[env_name] = major_score
                print('Save the model with %s env %s %0.4f' % (env_name, major_metric, major_score))
                speaker.save(idx_tobe, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s env loss %0.4f' % (env_name, loss))
                speaker.save(idx_tobe, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print(score_string)
        print()

        # Train for log_every interval
        speaker.env = train_env

        log_dict = defaultdict(lambda: 0)

        if idx_tobe <= pretrain_iters:
            ml_weight, policy_weight, baseline_weight = 1., 0., 3.
            log_dict = speaker.rl_train(reward_func, interval, ml_weight=ml_weight, policy_weight=policy_weight, baseline_weight=baseline_weight)
            if idx_tobe == pretrain_iters:
                speaker.save(idx_tobe,
                             os.path.join(log_dir, 'state_dict',
                                          'pretrain_iter%d_%0.3f_%0.3f_%0.3f' % (idx_tobe, ml_weight, policy_weight, baseline_weight)))
        else:
            rl_log = speaker.rl_train(reward_func, interval, ml_weight=0.05, policy_weight=1., baseline_weight=.5, entropy_weight=args.entropy,
                                      self_critical=args.self_critical
                                      )
            # rl_log = speaker.rl_train(reward_func, interval, ml_weight=1., policy_weight=0., baseline_weight=0., entropy_weight=args.entropy,
            #                           self_critical=args.self_critical
            #                           )
            for key, value in rl_log.items():
                log_dict[key + "/score_rl"] += value

        train_log_str = "Iter %05d, " % idx_tobe
        for name, value in log_dict.items():
            writer.add_scalar(name, value / interval, idx_tobe)
            train_log_str += "%s: %0.4f  " % (name, value / interval)
        print(train_log_str)

def train_gan_speaker(train_env, fake_env, neg_env, tok, n_iters, log_every=100, val_envs={}):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)
    arbiter = Arbiter(train_env, listner, tok)

    print("The size of training enviroment is %d" % train_env.size())
    print("The size of FAKE enviroment is %d" % fake_env.size())

    def train_arbiter_step():
        # Train GT      (train_path, train_inst)
        train_env.reset()
        train_batch = train_env.batch
        train_gt_loss = arbiter.train_gt(train_env, label=1.)

        # Train FAKE sampled from the train_env  (train_path, gen_inst)
        speaker.env = train_env
        train_env.reset(train_batch)            # Use the same batch as the real
        insts, _, _, _ = speaker.infer_batch(sampling=True, train=False)
        insts = [[tok.word_to_index['<BOS>']] + list(tok.shrink(inst)) + [tok.word_to_index['<EOS>']] for inst in insts]
        arbiter.env = train_env
        train_fake_loss = arbiter.train_fake(train_batch, insts)

        if args.add_neg:
            # Train GT2      (train_path, train_inst)
            train_env.reset()
            train_gt_loss += arbiter.train_gt(train_env, label=1.)
            train_gt_loss /= 2

            # Train Neg     (hard_neg_path, train_inst) and (train_path, hard_neg_inst)
            neg_env.reset()
            train_neg_loss = arbiter.train_gt(neg_env,   label=0.)
        else:
            train_neg_loss = 0.

        # Train FAKE sampled from the aug_env    (aug_path,   gen_inst)
        # speaker.env = fake_env
        # arbiter.env = fake_env
        # fake_env.reset()
        # fake_batch = fake_env.batch
        # insts, _, _ = speaker.infer_batch(sampling=True, train=False)
        # aug_fake_loss = arbiter.train_fake(fake_batch, insts)
        # aug_fake_loss = 0.

        return {'d_train_gt_loss':      train_gt_loss,
                'd_train_fake_loss':    train_fake_loss,
                'd_train_neg_loss':     train_neg_loss
                }

    # Add <BOS>, <EOS> to the instruction d discrete the answer with 0.5
    # Because the torch doesn't support np.bool_, convert it into uint8 first
    # reward_func = lambda batch, pathXinst: \
    #     (arbiter.infer_batch(batch, [[tok.word_to_index["<BOS>"]] + list(inst) + [tok.word_to_index["<EOS>"]]
    #                                 for _, inst in pathXinst]) > 0.5).astype(np.uint8)
    reward_func = lambda batch, pathXinst: \
        arbiter.infer_batch(batch, [[tok.word_to_index["<BOS>"]] + list(inst) + [tok.word_to_index["<EOS>"]]
                                     for _, inst in pathXinst]) - 0.5

    start_iter = 0
    pretrain_arbiter = 200
    pretrain_baseline = -100 if args.self_critical else 200
    if args.load is not None:
        print("LOAD THE DICT from %s" % args.load)
        if "pretrain_arbiter" in args.load:
            speaker.load(args.load)
        else:
            if "rl_gan" in args.load:           # Load the start_iter if it's trained with RL
                start_iter = speaker.load(args.load)
            else:                           # For pretrain model, directly load it
                speaker.load(args.load)

    if args.arbiter is not None:
        print("Load arbiter at iter %d" % arbiter.load(args.arbiter))

    if args.fast_train:
        log_every = 50
        pretrain_arbiter = 10
        pretrain_baseline = 0

    if args.load is None or ("pretrain_arbiter" not in args.load and "rl_gan" not in args.load):        # If the arbiter is loaded, do not train it
        # Converge the Arbiter first (with the initial speaker)
        if not args.fix_arbiter:
            for _ in tqdm.tqdm(range(pretrain_arbiter)):
                train_arbiter_step()
            speaker.save(0, os.path.join(log_dir, 'state_dict', 'pretrain_arbiter'))    # Save the speaker for pretrain arbiter

    assert pretrain_baseline % log_every == 0
    best_score = defaultdict(lambda: 0)
    for idx in range(start_iter, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        idx_tobe = idx + interval

        print("Iter: %d" % idx)

        # Evaluation and Save
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name:                                 # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()     # The dict here is to avoid multiple evaluation for one path
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            avg_length = utils.average_length(path2inst)
            pathXinst = list(path2inst.items())
            name2score = evaluator.lang_eval(pathXinst, no_metrics=('METEOR',))
            major_score = name2score['BLEU']
            score_string = " "
            for score_name, score in name2score.items():
                writer.add_scalar("lang_score/%s/%s" % (score_name, env_name), score, idx)
                score_string += "%s_%s: %0.4f " % (env_name, score_name, score)

            # Tensorboard log
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("avg_length/%s" % (env_name), avg_length, idx)

            # Save the model according to the bleu score
            if major_score > best_score[env_name]:
                best_score[env_name] = major_score
                print('Save the model with %s env bleu_4 %0.4f' % (env_name, major_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))
                arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu_arbiter' % env_name))

            if idx % 10000 == 0:
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'Iter_%06d' % idx))
                arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'Iter_%06d_arbiter' % idx))

            # Screen print out
            print(score_string)
        print()

        if idx_tobe <= pretrain_baseline:          # Prerain the baseline
            speaker.env = train_env
            arbiter.env = train_env
            ml_weight, policy_weight, baseline_weight = 1., 0., 5.
            log_dict = speaker.rl_train(reward_func, interval,
                                        ml_weight=ml_weight, policy_weight=policy_weight, baseline_weight=baseline_weight)
            print("Iter %d: pretrain baseline with weight, ml %0.4f, policy %0.4f, baseline %0.4f."
                  % (idx_tobe, ml_weight, policy_weight, baseline_weight))
            if idx_tobe == pretrain_baseline:
                speaker.save(idx_tobe,
                    os.path.join(log_dir, 'state_dict', 'pretrain_iter%d_%0.3f_%0.3f_%0.3f'
                                 % (idx_tobe, ml_weight, policy_weight, baseline_weight)))
        else:
            # GAN
            log_dict = defaultdict(lambda: 0.)
            arbiter_iters = interval // args.sdratio
            speaker_iters = arbiter_iters * args.sdratio
            if args.fix_arbiter:        # This is just for checking!!! Redundant checking!!
                arbiter_iters = 0
            speaker_idx = 0
            arbiter_idx = 0
            for _ in range(speaker_iters):
                # D_step
                if not args.fix_arbiter:
                    if speaker_idx % args.sdratio == 0:     # For every 'sdratio' speaker iters, run one arbiter iter
                        d_log = train_arbiter_step()
                        arbiter_idx += 1

                # G_step on train paths
                speaker.env = train_env
                arbiter.env = train_env
                g1_log = speaker.rl_train(reward_func, iters=1,
                                          ml_weight=args.ml_weight, policy_weight=1., baseline_weight=0.5, entropy_weight=0.,
                                          self_critical=args.self_critical
                                          )

                # G_step on fake paths
                speaker.env = fake_env
                arbiter.env = fake_env
                g2_log = speaker.rl_train(reward_func, iters=1,
                                          ml_weight=args.ml_weight, policy_weight=1., baseline_weight=0.5, entropy_weight=0.,
                                          self_critical=args.self_critical,
                                          ml_env=train_env
                                          )

                for key, value in g1_log.items():
                    log_dict[key + "/train"] += value
                for key, value in g2_log.items():
                    log_dict[key + "/fake"] += value
                if not args.fix_arbiter:
                    for key, value in d_log.items():
                        log_dict[key] += value
                speaker_idx += 1

            assert (arbiter_iters == arbiter_idx) and (speaker_iters == speaker_idx)
            train_log_str = "Iter %05d, speaker_iters %d, arbiter_iters %d" % (idx_tobe, speaker_iters, arbiter_iters)

            for name, value in log_dict.items():
                if "/train" in name or "/fake" in name:     # Calculating the average based on the number of iters
                    avg_value = value / speaker_iters
                else:
                    avg_value = value / arbiter_iters
                writer.add_scalar(name, avg_value, idx_tobe)
                train_log_str += "%s: %0.4f  " % (name, avg_value)

            print(train_log_str)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions, berkeley_bleu = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("berkeley_bleu/%s" % env_name, berkeley_bleu, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu, Berkeley_bleu, Loss, Word_Accu, Sent_Accu for %s is: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f" %
                  (env_name, bleu_score, berkeley_bleu, loss, word_accu, sent_accu))
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    ''' Train on training set, validating on both seen and unseen. '''
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    # agent.train(encoder_optimizer, decoder_optimizer, 1000, feedback='teacher')
    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40
    else:
        killer = utils.GracefulKiller()
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            # If I need to use advance dropout in training the listener
            make_noise = args.dropbatch or args.dropinstance or args.dropviewpoint
            listner.train(interval, feedback=feedback_method, make_noise=make_noise)   # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env
                    # NOTE: For here, if make_noise = None. It means that the listener will not do any dropout
                    # on the feature.
                    # args.ml_weight = 0.05
                    args.ml_weight = 0.1
                    listner.accumulate_gradient(feedback_method, make_noise=False)
                    listner.env = aug_env
                    # args.ml_weight = 0.2
                    args.ml_weight = 0.4        # Sem-Configuration
                    listner.accumulate_gradient(args.aug_feedback, make_noise=True, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train the ground truth
                    listner.env = train_env
                    args.ml_weight = 0.1
                    listner.train(1, feedback=feedback_method, make_noise=False, speaker=None)

                    # Train the aug_env
                    listner.env = aug_env
                    args.ml_weight = 0.4
                    listner.train(1, feedback=args.aug_feedback, make_noise=True, speaker=speaker)

        listner.timer.show()

        # Log the tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        # critic_loss = sum(listner.logs['critic_loss']) / length / args.batchSize
        # entropy = sum(listner.logs['entropy']) / length / args.batchSize
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Denoising log
        if args.denoise != 0:
            avg_func = lambda name: sum(listner.logs[name]) / max(len(listner.logs[name]), 1)
            denoise_log_str = ""
            for log_name in ("gate", "activate_gate", "close_gate", "teacher_actions", "avg_ml_loss"):
                writer.add_scalar(log_name, avg_func(log_name), idx)
                denoise_log_str += "%s %0.4f, " % (log_name, avg_func(log_name))
            print(denoise_log_str)

        loss_str = ""
        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280
            # listner.test(use_dropout=True, feedback='sample', allow_cheat=True, iters=iters)
            # val_losses = np.array(listner.losses)
            # val_loss_avg = np.average(val_losses)

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            # loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))


        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))


        if not args.fast_train:
            if killer.kill_now:
                break

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

def valid(train_env, tok, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def beam_valid(train_env, tok, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric,val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, args.iters, val_envs=val_envs)
        else:
            valid(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    elif args.train == 'rlspeaker':
        train_rl_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'ganspeaker':
        fake_env = SemiBatch(False, 'tasks/R2R/data/all_paths_46_removetrain.json',
                  feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train', 'val_seen'], tokenizer=tok)
        neg_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                           splits=['train_pathneg', 'train_instneg'], tokenizer=tok)
        train_gan_speaker(train_env, fake_env, neg_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'searchinst':
        listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        speaker = Speaker(train_env, listner, tok)
        speaker.load(args.speaker)
        speaker.search_inst()
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)


    for args.beam in [False, True]:
        print("Using Beam Search %s" % args.beam)
        for env_name, (env, evaluator) in val_envs.items():
            if env_name == 'train':
                continue
            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid(beam=args.beam, wrapper=tqdm.tqdm)
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[path_id]['instructions'])
            #bleu_score, precisions = evaluator.bleu_score(path2inst)
            pathXinst = list(path2inst.items())
            name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
            score_string = " "
            for score_name, score in name2score.items():
                score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
            print("For env %s" % env_name)
            print(score_string)
            print("Average Length %0.4f" % utils.average_length(path2inst))

def create_augment_data():
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load features
    feat_dict = read_img_features(features)
    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)

    # The datasets to be augmented
    print ("Start to augment the data")
    aug_envs = []
    aug_envs.append(
        SemiBatch(False, 'tasks/R2R/data/all_paths_46_removetrain.json',
            feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train', 'val_seen'], tokenizer=tok)
    )
    aug_envs.append(
        R2RBatch(
            feat_dict, candidate_dict, batch_size=args.batchSize, splits=['val_seen'], tokenizer=tok
        )
    )
    aug_envs.append(
        R2RBatch(
            feat_dict, candidate_dict, batch_size=args.batchSize, splits=['val_unseen'], tokenizer=tok
        )
    )

    for snapshot in os.listdir(os.path.join(log_dir, 'state_dict')):
        if snapshot != "best_val_unseen_bleu":  # Select a particular snapshot to process. (O/w, it will make for every snapshot)
            continue

        # Create Speaker
        listner = Seq2SeqAgent(aug_envs[0], "", tok, args.maxAction)
        speaker = Speaker(aug_envs[0], listner, tok)

        # Load Weight
        speaker.load(os.path.join(log_dir, 'state_dict', snapshot))

        # Augment the env from aug_envs
        for aug_env in aug_envs:
            speaker.env = aug_env

            # Create the aug data
            import tqdm
            path2inst = speaker.get_insts(beam=args.beam, wrapper=tqdm.tqdm)
            data = []
            for datum in aug_env.fake_data:
                datum = datum.copy()
                path_id = datum['path_id']
                if path_id in path2inst:
                    datum['instructions'] = [tok.decode_sentence(path2inst[path_id])]
                    datum.pop('instr_encoding')     # Remove Redundant keys
                    datum.pop('instr_id')
                    data.append(datum)

            print("Totally, %d data has been generated for snapshot %s." % (len(data), snapshot))
            print(datum)    # Print a Sample

            # Save the data
            import json
            os.makedirs(os.path.join(log_dir, 'aug_data'), exist_ok=True)
            json.dump(data,
                      open(os.path.join(log_dir, 'aug_data', '%s_%s.json' % (snapshot, aug_env.name)), 'w'),
                      sort_keys=True, indent=4, separators=(',', ': '))

def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)
    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    if args.aug is None:            # If aug is specified, load the "aug"
        speaker_snap_name = "adam_drop6_correctsave"
        print("Loading from %s" % speaker_snap_name)
        aug_path = "snap/speaker/long/%s/aug_data/best_val_unseen_loss.json" % speaker_snap_name
    else:   # Load the path from args
        aug_path = args.aug

    # The dataset used in training
    # splits = [aug_path, 'train'] if args.combineAug else [aug_path]

    # Create the training environment
    train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)

def train_arbiter(arbiter_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(arbiter_env, "", tok, args.maxAction)
    arbiter = Arbiter(arbiter_env, listner, tok)
    best_f1 = 0.
    best_accu = 0.
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        arbiter.env = arbiter_env
        arbiter.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, env in val_envs.items():
            print("............ Evaluating %s ............." % env_name)
            arbiter.env = env
            path2prob = arbiter.valid()
            path2answer = env.get_answer()
            true_positive  = len([1 for path in path2prob if path2prob[path] >= 0.5 and     path2answer[path]])
            false_positive = len([1 for path in path2prob if path2prob[path] <  0.5 and     path2answer[path]])
            false_negative = len([1 for path in path2prob if path2prob[path] >= 0.5 and not path2answer[path]])
            true_negative  = len([1 for path in path2prob if path2prob[path] <  0.5 and not path2answer[path]])
            true_accu      = true_positive  / (true_positive + false_positive)
            true_recall    = true_positive  / max((true_positive + false_negative), 1)
            true_f1        = 2 * (true_accu * true_recall) / max((true_accu + true_recall), 1)
            false_accu     = true_negative  / (true_negative + false_negative)
            writer.add_scalar("true_accu", true_accu, idx)
            writer.add_scalar("true_recall", true_recall, idx)
            writer.add_scalar("true_f1", true_f1, idx)
            writer.add_scalar("false_accu", false_accu, idx)

            if env_name == 'valid':
                if true_f1 > best_f1:
                    best_f1 = true_f1
                    print('Save the model with %s f1 score %0.4f' % (env_name, best_f1))
                    arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_f1' % env_name))

                if true_accu > best_accu:
                    best_accu = true_accu
                    print("Save the model with %s true accu %0.4f" % (env_name, best_accu))
                    arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_accu' % env_name))

            print("True Accu %0.4f, False Accu %0.4f" % (true_accu, false_accu))
            sys.stdout.flush()


def filter_arbiter(valid_env, aug_env, tok):
    import tqdm
    listner = Seq2SeqAgent(aug_env, "", tok, args.maxAction)
    arbiter = Arbiter(aug_env, listner, tok)

    # Load the model
    arbiter.load(args.load)

    # Create Dir
    os.makedirs(os.path.join(log_dir, 'arbiter_result'), exist_ok=True)

    # Get the prob for the validation env (may be used for determining the threshold)
    arbiter.env = valid_env
    valid_inst2prob = arbiter.valid(wrapper=tqdm.tqdm)
    json.dump(valid_inst2prob, open(os.path.join(log_dir, 'arbiter_result', 'valid_prob.json'), 'w'))

    # Get the prob of the augmentation data
    arbiter.env = aug_env
    aug_inst2prob = arbiter.valid(wrapper=tqdm.tqdm)
    json.dump(aug_inst2prob, open(os.path.join(log_dir, 'arbiter_result', 'aug_prob.json'), 'w'))

    # Create the Dataset
    data = [datum.copy() for datum in aug_env.data if aug_inst2prob[datum['instr_id']] > 0.5]
    for datum in data:
        datum['instructions'] = [datum['instructions']]
        datum.pop('instr_encoding')     # Remove the redundant components in the dataset
        datum.pop('instr_id')
    return data

def arbiter():
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    if args.fast_train:
        feat_dict = read_img_features(features_fast)
    else:
        feat_dict = read_img_features(features)

    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)

    gt_train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=['val_unseen'], tokenizer=tok)
    gt_valid_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=['val_seen'], tokenizer=tok)

    # gt_train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
    #                         splits=['val_seen_half1'], tokenizer=tok)
    # gt_valid_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
    #                         splits=['val_seen_half2'], tokenizer=tok)

    # Where to load the original data
    speaker_snap_name = "adam_drop6_correctsave" if args.speaker is None else args.speaker
    aug_data_name = "best_val_unseen_bleu_FAKE"
    aug_data_path = "snap/speaker/long/%s/aug_data/%s.json" % (
        speaker_snap_name,
        aug_data_name
    )

    # Where to save the splitted data
    saved_path = os.path.join(log_dir, 'aug_data')
    os.makedirs(saved_path, exist_ok=True)
    gen_train_path = os.path.join(saved_path, "%s_%s.json" % (aug_data_name, 'train'))
    gen_valid_path = os.path.join(saved_path, "%s_%s.json" % (aug_data_name, 'valid'))
    gen_test_path = os.path.join(saved_path, "%s_%s.json" % (aug_data_name, 'test'))

    if args.train == 'arbiter':
        # Load the augmented data
        print("\nLoading the augmentation data from path %s" % aug_data_path)
        aug_data = json.load(open(aug_data_path))
        print("The size of the augmentation data is %d" % len(aug_data))

        # Shuffle and split the data.
        print("Creating the json files ...")
        random.seed(1)
        random.shuffle(aug_data)
        train_size = gt_train_env.size() * 1        # The size of training data should be much larger
        valid_size = gt_valid_env.size()            # The size of the test data
        print("valid size is %d " % valid_size)
        gen_train_data = aug_data[:train_size]
        gen_valid_data = aug_data[train_size: (train_size+valid_size)]
        gen_test_data  = aug_data[train_size+valid_size:]

        # Create the json files
        json.dump(gen_train_data, open(gen_train_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        json.dump(gen_valid_data, open(gen_valid_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        json.dump(gen_test_data,  open(gen_test_path,  'w'), sort_keys=True, indent=4, separators=(',', ': '))
        print("Finish dumping the json files\n")

        # Load augmentation Envs
        gen_train_path = "snap/speaker/long/%s/aug_data/%s.json" % (
            speaker_snap_name,
            "best_val_unseen_bleu_val_unseen"
        )
        aug_train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_train_path], tokenizer=tok)
        aug_valid_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_valid_path], tokenizer=tok)
        print("Loading the generated data from %s with size %d" % (aug_data_path, aug_train_env.size()))

        # Create Arbiter Envs
        arbiter_train_env = ArbiterBatch(gt_train_env, aug_train_env, args.batchSize//2, args.batchSize//2,
                                         feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)
        print("The size of Training data in Arbiter is %d" % arbiter_train_env.size())
        arbiter_valid_env = ArbiterBatch(gt_valid_env, aug_valid_env, args.batchSize//2, args.batchSize//2,
                                         feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)

        print("The size of Validation data in Arbiter is %d" % arbiter_valid_env.size())
        train_arbiter(arbiter_train_env, tok, args.iters,
                      val_envs={
                          'train': arbiter_train_env,
                          'valid': arbiter_valid_env
                      })
    if args.train == 'filterarbiter':
        # Load the augmentation test env
        # aug_test_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_test_path], tokenizer=tok)
        aug_test_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[aug_data_path], tokenizer=tok)
        print("%d data is loaded to be filtered" % (aug_test_env.size()))
        filter_data = filter_arbiter(gt_valid_env, aug_test_env, tok)
        print("The size of the remaining data is %d" % len(filter_data))
        json.dump(filter_data,
                  open(os.path.join(log_dir, "aug_data/%s_filter.json" % (aug_data_name)), 'w'),
                  sort_keys=True, indent=4, separators=(',', ': ')
                  )

def finetune():
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    if args.fast_train:
        feat_dict = read_img_features(features_fast)
    else:
        feat_dict = read_img_features(features)

    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    print("The finetune data_size is : %d\n" % train_env.size())
    val_envs = {split: (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok)) for split in ['train', 'val_seen', 'val_unseen']}

    train(train_env, tok, args.iters, val_envs=val_envs)

def test():
    setup()

    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE DICT from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

def meta_filter():
    """
    Train the listener with the augmented data
    """
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    if args.fast_train:
        feat_dict = read_img_features(features_fast)
    else:
        feat_dict = read_img_features(features)
    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    if args.aug is None:            # If aug is specified, load the "aug"
        speaker_snap_name = "adam_drop6_correctsave"
        print("Loading from %s" % speaker_snap_name)
        aug_path = "snap/speaker/long/%s/aug_data/best_val_unseen_loss.json" % speaker_snap_name
    else:   # Load the path from args
        aug_path = args.aug

    # Create the training environment
    aug_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok)
    train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                       splits=['train@3333'], tokenizer=tok)
    print("The augmented data_size is : %d" % train_env.size())
    stats = train_env.get_statistics()
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen@133']}

    val_env, val_eval = val_envs['val_unseen@133']

    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    def filter_result():
        listner.env = val_env
        val_env.reset_epoch()
        listner.test(use_dropout=False, feedback='argmax')
        result = listner.get_results()
        score_summary, _ = val_eval.score(result)
        for metric,val in score_summary.items():
            if metric in ['success_rate']:
                return val

    listner.load(args.load)
    base_accu = (filter_result())
    print("BASE ACCU %0.4f" % base_accu)

    success = 0

    for data_id, datum in enumerate(aug_env.data):
        # Reload the param of the listener
        listner.load(args.load)
        train_env.reset_epoch(shuffle=True)

        listner.env = train_env

        # Train for the datum
        # iters = train_env.size() // train_env.batch_size
        iters = 10
        for i in range(iters):
            listner.env = train_env
            # train_env.reset(batch=([datum] * (train_env.batch_size // 2)), inject=True)
            train_env.reset(batch=[datum] * train_env.batch_size, inject=True)
            # train_env.reset()
            # train_env.reset()
            listner.train(1, feedback='sample', reset=False)
        # print("Iter %d, result %0.4f" % (i, filter_result()))
        now_accu = filter_result()
        if now_accu > base_accu:
            success += 1
        # print("RESULT %0.4f" % filter_result())
        print('Accu now %0.4f, success / total: %d / %d = %0.4f' % (now_accu, success, data_id+1, success / (data_id + 1)))


        # Evaluate (on val_unseen only)


if __name__ == "__main__":

    if args.train in ['speaker', 'ganspeaker', 'rlspeaker', 'listener', 'validspeaker', 'validlistener',
                      'searchinst']:
        train_val()
    elif args.train == 'augment':
        create_augment_data()
    elif args.train == 'auglistener':
        train_val_augment()
    elif args.train == 'finetune':
        finetune()
    elif 'arbiter' in args.train:
        arbiter()
    elif 'meta' in args.train:
        meta_filter()
    else:
        assert False



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

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, SemiBatch, ArbiterBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import IPython


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

if args.place365:
    features = PLACE365_FEATURES
    CANDIDATE_FEATURES = PLACE365_CANDIDATE_FEATURES
else:
    features = IMAGENET_FEATURES
    CANDIDATE_FEATURES = IMAGENET_CANDIDATE_FEATURES
features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'

feedback_method = args.feedback # teacher or sample

print(args)

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


def train(train_env, tok, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE DICT from %s" % args.load)
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
        listner.env = train_env
        listner.train(interval, feedback=feedback_method)   # Train interval iters
        # listner.timer.show()

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
            loss_str += ", %s" % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        listner.env = train_env

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 20000 == 0:
            import shutil
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))
            shutil.copy(os.path.join("snap", args.name, "state_dict", "best_val_unseen"),
                        os.path.join("snap", args.name, "state_dict", "best_val_unseen_%06d" % (iter)))

        if not args.fast_train:
            if killer.kill_now:
                break

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

def valid(train_env, tok, n_iters, log_every=100, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter % d" % agent.load(args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env
        # Get validation loss under the same conditions as training
        # iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280
        iters = None
        # agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True, iters=iters)
        # val_losses = np.array(agent.losses)
        # val_loss_avg = np.average(val_losses)
        # loss_str += ', %s loss: %.4f' % (env_name), val_loss_avg)
        # Get validation distance from goal under test evaluation conditions
        # agent.logs['circle'] = 0
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        # print("In env %s, the circle cases are %d(%0.4f)" % (env_name, agent.logs['circle'],
        #                                                      agent.logs['circle']*1./env.size()))
        result = agent.get_results()
        score_summary, _ = evaluator.score(result)
        loss_str = "%s: " % env_name
        for metric,val in score_summary.items():
            loss_str += ', %s: %.3f' % (metric, val)
        print(loss_str)


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

    if args.fast_train:
        feat_dict = read_img_features(features_fast)
    else:
        feat_dict = read_img_features(features)

    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict
    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
         )
         for split in ['val_seen', 'val_unseen', 'train']
        )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False

def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(os.path.join(log_dir, 'state_dict', 'best_val_seen_bleu'))
    # speaker.load(os.path.join(log_dir, 'state_dict', 'best_val_unseen_loss'))

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
            bleu_score, precisions, _ = evaluator.bleu_score(path2inst)
            print("Bleu, Loss, Word_Accu, Sent_Accu for %s is: %0.4f, %0.4f, %0.4f, %0.4f" %
                  (env_name, bleu_score, loss, word_accu, sent_accu))
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))
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
    # aug_envs.append(
    #     R2RBatch(
    #         feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok
    #     )
    # )
    # aug_envs.append(
    #     SemiBatch(False, 'tasks/R2R/data/all_paths_46_removetrain.json',
    #         feat_dict, candidate_dict, batch_size=args.batchSize, splits=['train', 'val_seen'], tokenizer=tok)
    # )
    aug_envs.append(
        SemiBatch(False, 'tasks/R2R/data/all_paths_46_removevalunseen.json', "unseen",
                  feat_dict, candidate_dict, batch_size=args.batchSize, splits=['val_unseen'], tokenizer=tok)
    )
    aug_envs.append(
        SemiBatch(False, 'tasks/R2R/data/all_paths_46_removetest.json', "test",
                  feat_dict, candidate_dict, batch_size=args.batchSize, splits=['test'], tokenizer=tok)
    )
    # aug_envs.append(
    #     R2RBatch(
    #         feat_dict, candidate_dict, batch_size=args.batchSize, splits=['val_seen'], tokenizer=tok
    #     )
    # )
    # aug_envs.append(
    #     R2RBatch(
    #         feat_dict, candidate_dict, batch_size=args.batchSize, splits=['val_unseen'], tokenizer=tok
    #     )
    # )

    for snapshot in os.listdir(os.path.join(log_dir, 'state_dict')):
        # if snapshot != "best_val_unseen_bleu":  # Select a particular snapshot to process. (O/w, it will make for every snapshot)
        if snapshot != "best_val_unseen_bleu":
            continue

        # Create Speaker
        listner = Seq2SeqAgent(aug_envs[0], "", tok, args.maxAction)
        speaker = Speaker(aug_envs[0], listner, tok)

        # Load Weight
        load_iter = speaker.load(os.path.join(log_dir, 'state_dict', snapshot))
        print("Load from iter %d"% (load_iter))

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
            print("Average Length %0.4f" % utils.average_length(path2inst))
            print(datum)    # Print a Sample

            # Save the data
            import json
            os.makedirs(os.path.join(log_dir, 'aug_data'), exist_ok=True)
            beam_tag = "_beam" if args.beam else ""
            json.dump(data,
                      open(os.path.join(log_dir, 'aug_data', '%s_%s%s.json' % (snapshot, aug_env.name, beam_tag)), 'w'),
                      sort_keys=True, indent=4, separators=(',', ': '))

def train_val_augment():
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

    # The dataset used in training
    splits = [aug_path, 'train'] if args.combineAug else [aug_path]

    # Create the training environment
    if args.half_half:
        assert args.aug is not None
        gt_env    = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                             splits=['train'], tokenizer=tok)
        aug_env   = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                             splits=[aug_path], tokenizer=tok)
        train_env = ArbiterBatch(gt_env, aug_env, args.batchSize//2, args.batchSize//2,
                                 feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)
    else:
        train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                         splits=splits, tokenizer=tok)

    print("The augmented data_size is : %d" % train_env.size())
    # stats = train_env.get_statistics()
    # print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    # print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs)

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
            if env_name == 'train' or env_name == 'val_unseen':
                path2prob = arbiter.valid(total=500)
            else:       # val_seen need accurate accuracy to evaluate the model performance (for early stopping)
                path2prob = arbiter.valid()
            print("len path2prob", len(path2prob))
            path2answer = env.get_answer()
            print("len path2ans", len(path2answer))
            false_probs = list([path2prob[path] for path in path2prob if not path2answer[path]])
            true_positive  = len([1 for path in path2prob if (path2prob[path] >= 0.5 and     path2answer[path])])
            false_positive = len([1 for path in path2prob if (path2prob[path] <  0.5 and     path2answer[path])])
            false_negative = len([1 for path in path2prob if (path2prob[path] >= 0.5 and not path2answer[path])])
            true_negative  = len([1 for path in path2prob if (path2prob[path] <  0.5 and not path2answer[path])])
            true_accu      = true_positive  / (true_positive + false_positive)
            true_recall    = true_positive  / max((true_positive + false_negative), 1)
            true_f1        = 2 * (true_accu * true_recall) / max((true_accu + true_recall), 1)
            false_accu     = true_negative  / (true_negative + false_negative)
            print("tp %d, fp %d, fn %d, tn %d" % (true_positive, false_positive, false_negative, true_negative))
            print("All negative", true_negative + false_negative)
            print("All positive", true_positive + false_positive)
            writer.add_scalar("true_accu", true_accu, idx)
            writer.add_scalar("true_recall", true_recall, idx)
            writer.add_scalar("true_f1", true_f1, idx)
            writer.add_scalar("false_accu", false_accu, idx)

            if env_name == 'val_seen':
                if true_f1 > best_f1:
                    best_f1 = true_f1
                    print('Save the model with %s f1 score %0.4f' % (env_name, best_f1))
                    arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_f1' % env_name))

                if true_accu > best_accu:
                    best_accu = true_accu
                    print("Save the model with %s true accu %0.4f" % (env_name, best_accu))
                    arbiter.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_accu' % env_name))

            print("True Accu %0.4f, False Accu %0.4f" % (true_accu, false_accu))
            print("Avg False probs %0.4f" % (sum(false_probs) / len(false_probs)))
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
    # arbiter.env = valid_env
    # valid_inst2prob = arbiter.valid(wrapper=tqdm.tqdm)
    # json.dump(valid_inst2prob, open(os.path.join(log_dir, 'arbiter_result', 'valid_prob.json'), 'w'))

    # Get the prob of the augmentation data
    arbiter.env = aug_env
    aug_inst2prob = arbiter.valid(wrapper=tqdm.tqdm)
    aug_data = [datum.copy() for datum in aug_env.data]
    for datum in aug_data:
        datum['instructions'] = [datum['instructions']]
        datum.pop('instr_encoding')     # Remove the redundant components in the dataset

    for datum in aug_data:
        datum['prob'] = aug_inst2prob[datum['instr_id']]
    json.dump(aug_data, open(os.path.join(log_dir, 'arbiter_result', 'aug_prob.json'), 'w'))

    # Create the Dataset
    data = [datum for datum in aug_data if aug_inst2prob[datum['instr_id']] > 0.5]

    for datum in aug_data:
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
                            splits=['train'], tokenizer=tok)
    gt_val_unseen_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                            splits=['val_unseen'], tokenizer=tok)
    gt_val_seen_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
                            splits=['val_seen'], tokenizer=tok)

    # gt_train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
    #                         splits=['val_seen_half1'], tokenizer=tok)
    # gt_valid_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize,
    #                         splits=['val_seen_half2'], tokenizer=tok)

    # Where to load the original data
    speaker_snap_name = "adam_drop6_correctsave" if args.speaker is None else args.speaker
    snapshot = "Iter_060000"
    aug_data_name = snapshot + "_FAKE"
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
        valid_size = gt_val_seen_env.size()            # The size of the test data
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
        gen_train_path = "snap/speaker/long/%s/aug_data/%s.json" % (        # Train: unseen generate vs unseen gt
            speaker_snap_name,
            snapshot + "_val_unseen"
        )
        aug_val_unseen_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_train_path], tokenizer=tok)

        aug_fake_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_valid_path], tokenizer=tok)

        gen_valid_path = "snap/speaker/long/%s/aug_data/%s.json" % (        # Valid:   seen generate vs   seen gt
            speaker_snap_name,
            snapshot + "_val_seen"
        )
        aug_val_seen_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_valid_path], tokenizer=tok)

        # gen_valid_path = "snap/speaker/long/%s/aug_data/%s.json" % (        # Valid:   seen generate vs   seen gt
        #     speaker_snap_name,
        #     snapshot + "_train"
        # )
        # aug_train_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_valid_path], tokenizer=tok)
        # print("Loading the generated data from %s with size %d" % (aug_data_path, aug_train_env.size()))

        # Create Arbiter Envs
        arbiter_train_env = ArbiterBatch(gt_val_unseen_env, aug_val_unseen_env, args.batchSize//2, args.batchSize//2,
                                         feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)
        print("The size of Training data in Arbiter is %d" % arbiter_train_env.size())
        arbiter_valid_env = ArbiterBatch(gt_val_seen_env, aug_val_seen_env, args.batchSize//2, args.batchSize//2,
                                         feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)

        arbiter_valid_fake_env = ArbiterBatch(gt_val_seen_env, aug_fake_env, args.batchSize//2, args.batchSize//2,
                                              feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)
        # arbiter_valid_train_env = ArbiterBatch(gt_val_seen_env, aug_train_env, args.batchSize//2, args.batchSize//2,
        #                                        feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)

        print("The size of Validation data in Arbiter is %d" % arbiter_valid_env.size())
        train_arbiter(arbiter_train_env, tok, args.iters,
                      val_envs={
                          'train':       arbiter_train_env,
                          'val_seen':       arbiter_valid_env,
                          'valid_fake':  arbiter_valid_fake_env,
                          # 'valid_train': arbiter_valid_train_env,
                      })
    if args.train == 'filterarbiter':
        # Load the augmentation test env
        # aug_test_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[gen_test_path], tokenizer=tok)
        aug_test_env = R2RBatch(feat_dict, candidate_dict, batch_size=args.batchSize, splits=[aug_data_path], tokenizer=tok)
        print("%d data is loaded to be filtered" % (aug_test_env.size()))
        filter_data = filter_arbiter(gt_val_seen_env, aug_test_env, tok)
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

def hard_negative():
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    if args.fast_train:
        feat_dict = read_img_features(features_fast)
    else:
        feat_dict = read_img_features(features)

    candidate_dict = utils.read_candidates(CANDIDATE_FEATURES)

    gt_train_env, gt_val_seen_env, gt_val_unseen_env = gt_envs = list(R2RBatch(
        feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok)
        for split in ['train', 'val_seen', 'val_unseen']
    )
    neg_train_env, neg_val_seen_env, neg_val_unseen_env = neg_envs = list(R2RBatch(
        feat_dict, candidate_dict, batch_size=args.batchSize, splits=[split+"_instneg", split+"_pathneg"], tokenizer=tok)
        for split in ['train', 'val_seen', 'val_unseen']
    )
    arbiter_train_env, arbiter_val_seen_env, arbiter_val_unseen_env = (ArbiterBatch(
        gt_env, neg_env, args.batchSize // 2, args.batchSize // 2,
        feat_dict, candidate_dict, batch_size=args.batchSize, splits=[], tokenizer=tok)
        for gt_env, neg_env in zip(gt_envs, neg_envs)
    )
    train_arbiter(arbiter_train_env, tok, args.iters,
                  val_envs={
                      'train':         arbiter_train_env,
                      'val_seen':      arbiter_val_seen_env,
                      'val_unseen':    arbiter_val_unseen_env,
                  })

if __name__ == "__main__":

    if args.train in ['speaker', 'listener', 'validspeaker', 'validlistener']:
        train_val()
    elif args.train == 'augment':
        create_augment_data()
    elif args.train == 'auglistener':
        train_val_augment()
    elif args.train == 'finetune':
        finetune()
    elif args.train == 'hardneg':
        hard_negative()
    elif 'arbiter' in args.train:
        arbiter()
    elif 'meta' in args.train:
        meta_filter()
    else:
        assert False


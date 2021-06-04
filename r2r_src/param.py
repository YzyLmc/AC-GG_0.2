import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from :
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--arbiter", default=None)

        self.parser.add_argument("--aug", default=None)
        self.parser.add_argument("--combineAug", action='store_const', default=False, const=True)

        # Speaker Model Config
        self.parser.add_argument("--feedForward", dest='feed_forward', action="store_const", default=False, const=True)
        self.parser.add_argument("--packlstm", dest='pack_lstm', action="store_const", default=False, const=True)

        # Listener Model Config
        self.parser.add_argument("--fixedSizeCtx", dest='fixed_size_ctx', type=int, default=0)
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--denoise", type=float, default=0.0)
        self.parser.add_argument("--gradGate", dest="grad_gate", action='store_const', default=False, const=True)
        self.parser.add_argument("--straightGate", dest="straight_gate", action='store_const', default=False, const=True)
        self.parser.add_argument("--ratioGate", dest="ratio_gate", action='store_const', default=False, const=True)
        self.parser.add_argument("--place365", action='store_const', default=False, const=True)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--traditionalDrop", dest='traditional_drop', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Semantic View Config
        self.parser.add_argument("--dropmask01", action='store_const', default=False, const=True)

        # Label 1000
        self.parser.add_argument("--topk", type=int, default=-1)

        # Dropout
        self.parser.add_argument("--dropbatch", action='store_const', default=False, const=True)
        self.parser.add_argument("--dropinstance", action='store_const', default=False, const=True)
        self.parser.add_argument("--dropviewpoint", action='store_const', default=False, const=True)
        self.parser.add_argument("--dropspeaker", action='store_const', default=False, const=True)
        self.parser.add_argument('--featdropout', type=float, default=0.3)
        self.parser.add_argument("--independentDrop", dest='independent_drop', action='store_const', default=False, const=True)
        self.parser.add_argument("--featBatchNorm", dest='feat_batch_norm', action='store_const', default=False, const=True)
        self.parser.add_argument("--shufflefeat", type=int, default=0)

        # RL configurations
        self.parser.add_argument("--gradBaseline", dest="grad_baseline", action="store_const", default=False, const=True)
        self.parser.add_argument("--selfCritical", dest="self_critical", action="store_const", default=False, const=True)
        self.parser.add_argument("--metric", default="Bleu_4", type=str)
        self.parser.add_argument("--entropy", default=0., type=float)
        self.parser.add_argument("--sameInBatch", dest="same_in_batch", action="store_const", default=False, const=True)
        self.parser.add_argument("--normalizeReward", dest='normalize_reward', action='store_const', default=False, const=True)

        # GAN configuration
        self.parser.add_argument("--addNeg", dest='add_neg', action='store_const', default=False, const=True)
        self.parser.add_argument("--fixArbiter", dest='fix_arbiter', action='store_const', default=False, const=True)
        self.parser.add_argument("--sdratio", type=int, default=1)

        # SSL configuration
        self.parser.add_argument("--halfHalf", dest='half_half', action='store_const', default=False, const=True)
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)
        self.parser.add_argument("--augFeedback", dest="aug_feedback", type=str, default="sample")

        # Different Amount of Data:
        self.parser.add_argument("--filter", type=str, default="")

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam",action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift
        self.parser.add_argument("--elmo", action="store_const", default=False, const=True)
        self.parser.add_argument("--gumbel", action="store_const", default=False, const=True)
        self.parser.add_argument("--coverage", action="store_const", default=False, const=True)
        self.parser.add_argument("--eosub", action="store_const", default=False, const=True)
        self.parser.add_argument("--load", type=str, default=None)

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')

# The semantic category should not be used in dropout
# 0:Void, 1:Wall, 2:Floor, 4:Door, 17:Ceiling, 41:Unlabeled
NODROPCAT = [0, 1, 2, 4, 17, 41]

if args.topk != -1:
    import json
    import numpy as np
    LABEL1000_TOPK = np.array(json.load(open("models/sort_label1000.json"))[:args.topk])
    print("Do not drop top %d classes" % args.topk)


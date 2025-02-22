import json
import random
import time
import string
import os
import os.path as osp
from json import encoder
import numpy as np

def language_level(preds, metric=None):
    """

    :param preds: pred in MSCOCO type, [{'caption':'a sentence', 'image_id': the id}...]
    :param metric:
    :return:
    """
    import sys

    path_now, _ = os.path.split(os.path.realpath(__file__))
    sys.path.append(osp.join(path_now, "coco-caption"))
    annFile = osp.join(path_now, "annotation/navigation.json")
    from .cococaption.pycocotools.coco import COCO
    from .cococaption.pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    coco = COCO(annFile)
    valids = coco.getImgIds()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print(('using %d/%d predictions' % (len(preds_filt), len(preds))))
    # print preds_filt
    json.dump(preds_filt, open(tmp_name + '.json', 'w'))
    resFile = tmp_name + '.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate(metric=metric)

    os.system('rm ' + tmp_name + '.json')

    out = {}
    for metric, score in list(cocoEval.eval.items()):
        out[metric] = score

    return out

class Eval:
    def __init__(self):
        pass

    def eval(self, gt, pred):
        pass

    def eval_batch(self, gt, pred, real=True):
        assert gt.shape[0] == pred.shape[0]
        batch_size = gt.shape[0]
        result = np.zeros([batch_size], np.float32)
        for i in range(batch_size):
            result[i] = self.eval(gt, pred)
        return result


class LanguageEval(Eval):
    def __init__(self):
        from .cococaption.pycocoevalcap.evil import COCOEvilCap
        self.cocoEvil = COCOEvilCap()

    def eval_whole(self, gt, pred, **kwargs):
        import copy
        self.cocoEvil.evaluate(gt, pred, **kwargs)
        return copy.copy(self.cocoEvil.eval)

    def eval_batch(self, gt, pred, metric=None):
        """
        metric:
        :param gt:
        :param pred:
        :param metric: one of [Bleu_1, ..., Bleu_4, METEOR, ROUGE_L, CIDEr]
        :return:
        """
        self.cocoEvil.evaluate(gt, pred, {metric})
        result = np.zeros(len(gt), np.float32)
        for i in list(self.cocoEvil.imgToEval.keys()):
            result[i] = self.cocoEvil.imgToEval[i][metric]
        return result


class ActionEval(Eval):
    def __init__(self):
        Eval.__init__(self)

    def eval(self, gt, pred):
        pass

class DestinationEval(Eval):
    def __init__(self):
        Eval.__init__(self)

    def eval(self, gt, pred):
        if np.array_equal(gt[:2], pred[:2]):
            return 1.0
        else:
            return 0.0



if __name__ == "__main__":
    language_level([])
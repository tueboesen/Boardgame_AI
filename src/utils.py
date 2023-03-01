import glob
import os
import random

import numpy as np
import torch


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AttrDict(dict):
    """
    A dictionary that acts as an attribute so you can write dict_instance.key rather than dict_instance['key']
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def fix_seed(seed: int, include_cuda: bool = True) -> None:
    """
    Set the seed in order to create reproducible results, note that setting the seed also does it for gpu calculations, which slows them down.
    :param seed: an integer to fix the seed to
    :param include_cuda: whether to fix the seed for cuda calculations as well
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_file(folder,filename,ext):
    if filename == '':
        files = sorted(glob.glob(f'{folder}/*{ext}'))
        if len(files) > 0:
            file = files[-1]
    else:
        file = os.path.join(folder, f'{filename}{ext}')
    return file

def rand_argmax(tens):
    max_inds, = torch.where(tens == tens.max())
    return np.random.choice(max_inds)
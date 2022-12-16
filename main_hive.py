import logging
import random

import coloredlogs
import numpy as np
import torch

from Coach import Coach
from hive.HiveGame import HiveGame
from hive.nn.NNet import NNetWrapper
from utils import *
import sys
sys.setrecursionlimit(10000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

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

args = dotdict({
    'numIters': 1000,
    'numEps': 2,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    # positions = torch.arange(20).repeat(2).view(-1, 2)
    # positions[0,0] = 1
    # xy_dst1 = torch.tensor((1, 7))
    # xy_dst2 = torch.tensor((4, 5))
    # positions == xy_dst1  # should give none
    # positions == xy_dst2  # should give index 2 and 12
    #
    # def check(positions, xy):
    #     return (positions == xy_dst1.view(1, 2)).all(dim=1).nonzero()
    #
    # print(check(positions, xy_dst1))
    # # Output: tensor([], size=(0, 1), dtype=torch.int64)
    #
    # print(check(positions, xy_dst2))
    # # Output:
    # # tensor([[ 2],
    # #         [12]])

    log.info('Loading %s...', HiveGame.__name__)
    g = HiveGame()

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    fix_seed(1334)
    main()

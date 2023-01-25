import logging
import random

import coloredlogs
import numpy as np
import pygame
import torch

from Arena import Arena
from Coach import Coach
from hive.HiveGame import HiveGame
from hive.nn.NNet import NNetWrapper
from hive.ui import UI
from players import HumanPlayer
from utils import *
import sys
sys.setrecursionlimit(1000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 50,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'opp': 'random'
})


def main():

    log.info('Loading %s...', HiveGame.__name__)
    g = HiveGame()
    pygame.init()
    pygame.display.set_caption("Hive")
    display = UI(g)

    if args['opp'] == 'nn':
        log.info('Loading %s...', NNetWrapper.__name__)
        nnet = NNetWrapper(g)
        assert args.load_model
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    p1 = HumanPlayer(display)
    p2 = HumanPlayer(display)
    players = [p1,p2]
    A = Arena(players,g,display)
    A.single_game()



if __name__ == "__main__":
    main()

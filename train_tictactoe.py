import logging
import random

import coloredlogs
import numpy as np
import torch

from src.coach import Coach
from src.config import nn_args, coach_args, mcts_args
from src.nnet import NNetWrapper
from src.utils import *
import sys

from tictactoe.nn.ttt_nn import TicTacToeNNet
from tictactoe.ttt_game import TicTacToeGame
from tictactoe.ttt_ui import TicTacToeUI

sys.setrecursionlimit(1000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.



def main():
    log.info('Loading %s...', TicTacToeGame.__name__)
    g = TicTacToeGame()
    display = TicTacToeUI(g)
    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = TicTacToeNNet(g,nn_args)
    model = NNetWrapper(g,nnet,nn_args)

    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        nnet.load_checkpoint(coach_args.load_folder_file[0], coach_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, model, coach_args,display=display, mcts_args=mcts_args)

    if coach_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    fix_seed(1334)
    main()

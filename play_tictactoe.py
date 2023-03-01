import logging

import coloredlogs
import pygame

from src.arena import Arena
from src.players import RandomPlayer, HumanPlayer
from src.utils import *
import sys

from tictactoe.ttt_game import TicTacToeGame
from tictactoe.ttt_viz import TicTacToeViz

sys.setrecursionlimit(1000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = AttrDict({
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

    log.info('Loading %s...', TicTacToeGame.__name__)
    g = TicTacToeGame()
    display = TicTacToeViz(g)
    # p1 = HumanPlayer(display)
    p1 = RandomPlayer(display)
    p2 = RandomPlayer(display)
    players = [p1,p2]
    A = Arena(players,g,display)
    A.playGame()



if __name__ == "__main__":
    main()

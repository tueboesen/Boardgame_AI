import logging

import coloredlogs

from hive.hive_conf import hive_mcts_args, hive_nn_args
from hive.hive_game import HiveGame
from hive.nn.hive_nn import HiveNNet
from src.arena import Arena
from hive.hive_ui import UI, HiveUI
from src.mcts import MCTS
from src.nnet import NNetWrapper
from src.players import RandomPlayer, HumanPlayer, MCTSNNPlayer
from src.utils import *
import sys
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

    log.info('Loading %s...', HiveGame.__name__)
    g = HiveGame()
    display = HiveUI(g)

    nn = HiveNNet(g,hive_nn_args)
    nnet = NNetWrapper(g,nn,hive_nn_args)
    mcts1 = MCTS(nnet,hive_mcts_args)
    mcts2 = MCTS(nnet,hive_mcts_args)
    p1 = HumanPlayer(display)
    # p2 = HumanPlayer(display)
    # p1 = RandomPlayer(display)
    # p2 = RandomPlayer(display)
    # p1 = MCTSNNPlayer(mcts1)
    p2 = MCTSNNPlayer(mcts2)

    players = [p1,p2]
    A = Arena(players,g,display)
    n = 50
    A.playGames(n)



if __name__ == "__main__":
    main()

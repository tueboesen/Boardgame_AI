"""
A list of all the games available and references to all the files and classes needed.

Each game is represented by a dictionary.
All games need a "game" entrance which gives the rules of the game, following the template Game.py.
Theoretically that is all you need in order for you game to be played,
however the only available playmode at this point would be random, and without any vizualization
it would likely be hard to see whats going on.

For vizualization "viz" is required, following the template vizualization.py
For Human players "ui" is required, following the template ui.py.
For MCTSNN players "nn" is required with at least one configuration dictionary (if two is provided the second one is for the MCTS algorithm).

"""
from hive.hive_conf import hive_nn_args, hive_mcts_args, hive_coach_args
from hive.hive_game import HiveGame
from hive.hive_ui import HiveUI
from hive.nn.hive_nn import HiveNNet
from tictactoe.nn.ttt_nn import TicTacToeNNet
from tictactoe.ttt_conf import ttt_nn_args, ttt_mcts_args, ttt_coach_args
from tictactoe.ttt_game import TicTacToeGame
from tictactoe.ttt_viz import TicTacToeViz

TicTacToe = {
    'name': 'TicTacToe',
    'game': [TicTacToeGame],
    'nn': [TicTacToeNNet,ttt_nn_args,ttt_mcts_args,ttt_coach_args],
    'viz': [TicTacToeViz],
}

Hive = {
    'name': 'Hive',
    'game': [HiveGame],
    'nn': [HiveNNet,hive_nn_args,hive_mcts_args,hive_coach_args],
    'viz': [HiveUI],
}


GAMES_AVAILABLE = [TicTacToe,Hive]


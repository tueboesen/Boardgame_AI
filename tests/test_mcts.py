"""
This is a unit test of mcts.py
"""
import torch

from hive.hive_game import HiveGame
from hive.nn.hive_nn import HiveNNet
from src.mcts import MCTS, Node
import pytest

from src.nnet import NNetWrapper
from tictactoe.nn.ttt_nn import TicTacToeNNet
from tictactoe.ttt_conf import ttt_mcts_args, ttt_nn_args
from tictactoe.ttt_game import TicTacToeGame

# @pytest.fixture(params=[(TicTacToeGame(),TicTacToeNNet), (HiveGame(), HiveNNet)])
@pytest.fixture(scope="module", params=[(TicTacToeGame(),TicTacToeNNet)])
def mcts_node(request):
    g,nn_fnc = request.param
    nnet = nn_fnc(g,ttt_nn_args)
    model = NNetWrapper(g,nnet,ttt_nn_args)
    mcts_instance = MCTS(model, ttt_mcts_args)
    node = Node(g)
    return mcts_instance, node

class TestMcts:

    @pytest.fixture(autouse=True, scope='class')
    def setup(self,mcts_node):
        self.action_idx = 0
        self.mcts, self.node = mcts_node
        self.mcts.add_node(self.node)
        self.child_node = self.mcts.select_childnode(self.node, self.action_idx)

    def test_child_nodes_are_unique(self):
        child_node2 = self.mcts.select_childnode(self.node, self.action_idx)
        assert self.child_node != child_node2, "child nodes are not unique when they should be"

    def test_child_nodes_are_identical(self,):
        self.mcts.add_node(self.child_node, self.node.id, self.action_idx)
        child_node2 = self.mcts.select_childnode(self.node, self.action_idx)
        assert self.child_node == child_node2, "accessing the same child node that we previously added, should not create a new node"

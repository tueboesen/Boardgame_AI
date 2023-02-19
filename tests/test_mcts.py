"""
This is a unit test of mcts.py
"""
import torch

from hive.hive_game import HiveGame
from hive.nn.hive_nn import HiveNNet
from src.config import mcts_args, nn_args
from src.mcts import MCTS, Node
import pytest

from src.nnet import NNetWrapper
from tictactoe.nn.ttt_nn import TicTacToeNNet
from tictactoe.ttt_game import TicTacToeGame

@pytest.fixture
def game_ttt():
    g = TicTacToeGame()
    return g

@pytest.fixture
def node_ttt():
    g = TicTacToeGame()
    node = Node(g)
    return node

@pytest.fixture
def mcts_ttt():
    g = TicTacToeGame()
    nnet = TicTacToeNNet(g,nn_args)
    model = NNetWrapper(g,nnet,nn_args)
    mcts_instance = MCTS(model, mcts_args)
    return mcts_instance

# @pytest.fixture
# def example_hive():
#     g = HiveGame()
#     nnet = HiveNNet(g,nn_args)
#     model = NNetWrapper(g,nnet,nn_args)
#     mcts = MCTS(model, mcts_args)
#     return mcts


@pytest.mark.parametrize("mcts, node, action_idx", [
    (mcts_ttt, node_ttt, 0),
])


def test_select_childnode(mcts, node, action_idx):
    """
    Assert that childnode is creating a new node when needed.
    Assert that childnode is not creating a new node when not needed.
    """
    mcts.nodes[node.id] = node
    child_node = mcts.select_childnode(node, action_idx)
    child_node2 = mcts.select_childnode(node, action_idx)
    assert child_node != child_node2, "child nodes are not unique when they should be"
    mcts.add_node(child_node, node.id, action_idx)

    child_node2 = mcts.select_childnode(node, action_idx)
    assert child_node == child_node2, "accessing the same child node that we previously added, should not create a new node"



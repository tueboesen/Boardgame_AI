"""
MCGS implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import copy
import math

import torch
import numpy as np


mcts_config = {
    "puct_coefficient": 1.0,
    "num_simulations": 30,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,}

class MCGS:
    """
    The Monte Carlo Graph Search algorithm.

    """

    def __init__(self, game, model, mcts_param=mcts_config):
        self.model = model
        self.game = game
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

    def reset(self):
        pass

    def update_node(self,actionhist):
        """
        This routine is used when a MCTS-player plays with other players in a game.
        self.compute_action will perform a MCTS and select the most favorable action and advance the node to this state,
        but when other players performs a move we need some way to tell the MCTS-player that the game has advanced to a new state.
        This routine serves that purpose.

        actions should be a list of actions needed to propagate the game from the state saved in self.node, to the desired state.
        """
        return
    def compute_action(self):
        """
        """

        return

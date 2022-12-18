"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import torch
import numpy as np


class Node:
    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        self.game = parent.game
        self.action = action  # Action used to go to this state
        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.valid_actions = self.game.board.get_valid_moves()
        self.number_of_actions = len(self.valid_actions)
        self.child_total_value = torch.zeros([self.number_of_actions])  # Q
        self.child_priors = torch.empty([self.number_of_actions])  # P
        self.child_number_visits = torch.zeros([self.number_of_actions])  # N
        # self.valid_actions = obs["action_mask"].astype(np.bool)

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = self.game.board.rep_nn()

        self.mcts = mcts

    def __repr__(self) -> str:
        return f"Turn={self.board.turn},player={'white' if self.board.whites_turn else 'black'}"


    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        idx = torch.argmax(child_score)
        return self.valid_actions[idx]

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors.cpu()

    def get_child(self, action):
        if action not in self.children:
            self.game.set_state(self.state)
            obs, reward, done, _ = self.game.step(action)
            next_state = self.game.get_state()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts,
            )
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent


class RootParentNode:
    def __init__(self, game):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.game = game


mcts_config = {
    "puct_coefficient": 1.0,
    "num_simulations": 30,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,}

class MCTS:
    def __init__(self, game, model, args=None,mcts_param=mcts_config):
        self.model = model
        self.game = game
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.predict(leaf.obs)
                valids = leaf.valid_actions
                child_priors = child_priors[valids]
                child_priors = child_priors / child_priors.sum()
                # if self.add_dirichlet_noise:
                #     child_priors = (1 - self.dir_epsilon) * child_priors
                #     torch.ones_like(child_priors)*self.dir_noise
                #     child_priors += self.dir_epsilon * torch.distributions.dirichlet.Dirichlet(torch.ones_like(child_priors)*self.dir_noise).samples()

                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy
        )  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]

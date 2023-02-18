"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import copy
import math

import torch
import numpy as np


class Node:
    """
    A Tree node that contains the current board state, the possible actions from this state (child states), the number of visits to each of those as well as their expected outcomes.
    """
    def __init__(self, game):
        self.game = game
        self.id = game.canonical_string_rep()
        self.valid_actions = game.get_valid_moves()
        self.game_over = game.game_over
        self.value = game.reward[game.current_player]
        self.nn_rep = game.nn_rep()
        self.child_prior = None
        self.current_player = game.current_player
        self.child_values = torch.zeros(len(self.valid_actions),dtype=torch.float32)
        self.child_visits = torch.zeros_like(self.valid_actions)
        self.child_rep = {}

    def __repr__(self):
        return self.id
    def expand(self,child_priors):
        self.child_prior = child_priors.cpu()






mcts_config = {
    "puct_coefficient": 1.41,
    "num_simulations": 20,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,}

class MCTS:
    """
    The Monte Carlo Tree Search algorithm.

    This algorithm works on a tree structure governed by Node.
    When compute_action is

    Note that this algorithm
    """

    def __init__(self, model, mcts_param=mcts_config,rep=None,play_to_win=False):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = play_to_win
        self.decay = 0.9
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
        self.nodes = {}
        self.rep = rep
        self.tmp_hist = []

    def __repr__(self):
        return f"{self.rep}"

    def __str__(self):
        return f"{self.rep}"

    def reset(self):
        # node = Node(mcts=self, action=None, game_over=False, reward=0, game=self.game, parent=RootParentNode(self.game))
        self.nodes = {}

    def find_leaf(self,node):
        self.tmp_hist = []
        leaf = None
        while leaf is None:
            s = node.id
            if s in self.nodes and (not self.nodes[s].game_over):
                node = self.nodes[s]
                action_idx = self.select_action_idx(node)
                node = self.select_childnode(node,action_idx)
                self.tmp_hist.append((s,action_idx,node.id))
            else:
                leaf = node
        return leaf

    def select_action_idx(self,node):
        Q = node.child_values / (1 + node.child_visits)
        U = math.sqrt(node.child_visits.sum()+1)* node.child_prior/ (1 + node.child_visits)
        child_score = Q + self.c_puct * U
        idx = torch.argmax(child_score).item()
        return idx

    def select_childnode(self, node, action_idx):
        if action_idx in node.child_rep:
            s = node.child_rep[action_idx]
            node = self.nodes[s]
        else:
            game = node.game
            game_next = copy.deepcopy(game)
            action = node.valid_actions[action_idx].item()
            game_next.perform_action(action)
            node = Node(game_next)
        return node


    def backup(self,value,cp):
        for i, (s,a,s_next) in enumerate(reversed(self.tmp_hist)):
            node = self.nodes[s]
            if node.current_player == cp:
                sign = 1
            else:
                sign = -1
            node.child_values[a] += self.decay**i * sign * value
            node.child_visits[a] += 1
            node.child_rep[a] = s_next
        return

    def compute_action(self,game):
        """
        When called, this routine performs a number of simulation equal to self.num_sims.
        Each simulation starts from the game state defined in self.node and explores from that state until a leaf node is found (a node that has not previously been explored)
        The leaf node is then evaluated using the model (typically a neural network), and the policy vector and its value is stored in the nodes.
        After the specified number of simulations have been performed, the recommended action_idx is returned along with the probabilities and the child node.
        """
        s = game.canonical_string_rep()
        if s in self.nodes:
            node = self.nodes[s]
        else:
            node = Node(game)
        for _ in range(self.num_sims):
            leaf = self.find_leaf(node)
            if leaf.game_over:
                value = leaf.value
                # value = reward[leaf.game.previous_player]
                # if value > 0:
                #     leaf.parent.child_game_over[parent_action] = True
            else:
                child_priors, value = self.model.predict(leaf.nn_rep)
                valids = leaf.valid_actions
                child_priors = child_priors[valids]
                child_priors = child_priors / child_priors.sum()
                # if self.add_dirichlet_noise:
                #     child_priors = (1 - self.dir_epsilon) * child_priors
                #     torch.ones_like(child_priors)*self.dir_noise
                #     child_priors += self.dir_epsilon * torch.distributions.dirichlet.Dirichlet(torch.ones_like(child_priors)*self.dir_noise).samples()
                leaf.expand(child_priors)
            if leaf.id not in self.nodes:
                self.nodes[leaf.id] = leaf

            self.backup(value,leaf.game.current_player)


        # Tree policy target (TPT)
        # winning_moves = node.child_game_over.nonzero()
        # if len(winning_moves) > 0:
        #     action_idx = winning_moves[0].item()
        #     tree_policy = torch.zeros_like(node.child_number_visits)
        #     tree_policy[action_idx] = 1.0
        # else:
        tree_policy = node.child_visits / node.child_visits.sum()
        # tree_policy = torch.nn.functional.softmax(tree_policy/self.temperature,dim=0)
        # tree_policy = tree_policy / torch.max(tree_policy)  # to avoid overflows when computing softmax
        # tree_policy = np.power(tree_policy, self.temperature)
        # tree_policy = tree_policy / torch.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action_idx that has the maximum
            # tree policy probability
            # action_idx = torch.argmax(tree_policy).item()
            action_indices = (tree_policy == tree_policy.max()).nonzero()[:,0]
            a = torch.randint(action_indices.shape[0],(1,))
            action_idx = action_indices[a].item()
        else:
            # otherwise sample an action_idx according to tree policy probabilities
            action_idx = np.random.choice(np.arange(len(node.valid_actions)), p=tree_policy.numpy())
        node_next = self.select_childnode(node,action_idx)
        action = node.valid_actions[action_idx].item()
        return tree_policy, action, node_next.game

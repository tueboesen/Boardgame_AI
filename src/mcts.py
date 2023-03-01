"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import copy
import glob
import math
import os
from typing import Optional

import torch
import numpy as np

from hive.hive_utils import draw_board
from src.utils import get_file, rand_argmax


class Node:
    """
    A Tree node that contains the current board state, the possible actions from this state (child states), the number of visits to each of those as well as their expected outcomes.
    """
    def __init__(self, game):
        self.game = copy.deepcopy(game)
        # self.game_backup = copy.deepcopy(game)
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
        cls = self.__class__.__name__
        return f'{cls} id={self.id}, game={self.game!r}'

    def expand(self,child_priors):
        """
        Adds prior information to the node, which means the node has been expanded.
        """
        self.child_prior = child_priors.cpu()





class MCTS:
    """
    The Monte Carlo Tree Search algorithm.

    This algorithm uses nodes that are identified by their unique canonical representation of the game state they represent.

    Note that we distinguish between action and action_idx, action is a number between 0 and game.actionsize, while action_idx is a number between 0 and game.get_valid_actions()
    So action is the unique identifier of the action taken, while action_idx is the relative identifier of the action among the actions available.
    action = game.get_valid_actions()[action_idx]

    """

    def __init__(self, model, mcts_param, description=None, play_to_win=False):
        self.mcts_param = mcts_param
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = play_to_win
        self.decay = mcts_param['decay']
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
        self.nodes = {}
        self.description = description
        self.use_dummy_nn = False
        # self.tmp_hist = [] # should contain elements of type (s, a, s_next), where s is the canonical representation of a node, a is the action from that node which brings you to s_next.

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls} (description={self.description!r}, model={self.model!r}, mcts_param={self.mcts_param!r}, play_to_win={self.exploit!r})'

    def __str__(self):
        return f"{self.description}"

    def reset(self):
        """
        Cleans all the stored nodes from the MCTS algorithm
        """
        self.nodes = {}

    def save(self,folder,filename,ext='.mcts'):
        file = os.path.join(folder,f"{filename}{ext}")
        torch.save(self.nodes,file)
        return

    def load(self,folder,filename='',ext='.mcts'):
        file = get_file(folder, filename, ext)
        if not os.path.exists(file):
            raise (f"No model in path {file}")
        self.nodes = torch.load(file)


    def find_leaf(self,node : Node) -> (Node, list):
        """
        Starting from node, it searches through the tree structure until it finds a leaf node (meaning a node that hasn't been explored before, or that ends the game)
        This is the selection part of the MCTS algorithm.
        Note that self.tmp_hist is cleaned at the start of this and filled up with information about all the nodes that it goes through
        """
        hist = []
        leaf = None
        while leaf is None:
            s = node.id
            if s in self.nodes and (not self.nodes[s].game_over):
                node = self.nodes[s]
                visited = self.detect_previous_visit(node,hist)
                action_idx = self.select_action_idx(node, visited)
                node = self.select_childnode(node,action_idx)
                hist.append((s,action_idx,node.id))
            else:
                leaf = node
        return leaf, hist

    def detect_previous_visit(self,node: Node,hist: list) -> torch.LongTensor:
        """
        This function detects whether the node has previously been visited during this search.
        This is important since we want to prevent an infinite loop during the find leaf phase.
        This will only happen in games where actions can repeat themselves, and where the representation does not include turns
        """
        s = node.id
        v = torch.zeros_like(node.child_visits)
        for (n,action,_) in hist:
            if n == s:
                v[action] += 1
        return v

    def select_action_idx(self,node: Node, visited: Optional[torch.LongTensor]=None) -> int:
        """
        Selects an action_idx based on the standard MCTS selection criteria, which balances exploration and exploitation.
        """
        Q = node.child_values / (1 + node.child_visits)
        U = math.sqrt(node.child_visits.sum()+1)* node.child_prior/ (1 + node.child_visits)
        child_score = Q + self.c_puct * U
        if visited is not None:
            child_score /= (visited + 1)
        idx = int(rand_argmax(child_score))
        # idx = torch.argmax(child_score).item()
        return idx

    def select_childnode(self, node: Node, action_idx: int) -> Node:
        """
        Given a node and an action_idx it selects/creates the node_next this action_idx leads to when applied to node.
        """
        assert isinstance(action_idx, int) # We only accept int here since torch.tensors do not work with dictionary lookup.
        if action_idx in node.child_rep: # Node already exist
            s = node.child_rep[action_idx]
            node = self.nodes[s]
        else: # Node does not exist, we create the node
            game = node.game
            game_next = copy.deepcopy(game)
            action = node.valid_actions[action_idx].item()
            assert (node.valid_actions == game.get_valid_moves()).all()
            game_next.perform_action(action)
            node = Node(game_next)
            assert (game_next.get_valid_moves() == node.valid_actions).all()
        return node

    def backup(self,value: torch.FloatTensor,cp: int, hist: list):
        """
        Backpropagates the value to all the nodes stored in hist.
        cp designates the player who got the value, for all other players we currently assume -value

        Note that this method uses self.decay, which requires hist to be ordered (from start_node to finish_node)
        If self.decay < 1 then the value that we are propagating diminishes as it is propagated back through the nodes.
        This makes sense since the move that lead to value, becomes less valuable the more moves you go back.
        """
        for i, (s,a,s_next) in enumerate(reversed(hist)):
            node = self.nodes[s]
            if node.current_player == cp:
                sign = 1
            else:
                sign = -1
            node.child_values[a] += self.decay**i * sign * value
            node.child_visits[a] += 1
        return

    def add_node(self,node: Node,parent_string: Optional[str]=None, action_idx: Optional[int]=None):
        """
        Adds a node to self.nodes, and establishes its connection with its parent.
        """
        self.nodes[node.id] = node
        if (parent_string is not None) and (action_idx is not None):
            parent_node = self.nodes[parent_string]
            parent_node.child_rep[action_idx] = node.id
        return

    def compute_action(self,game):
        """
        When called, this routine performs a number of simulation equal to self.num_sims.
        Each simulation starts from the node associated with the input game state.
        From this node a leaf node is found.
        The leaf node is then evaluated using the model (typically a neural network), and the policy vector and its value is stored in the nodes.
        After the specified number of simulations have been performed, the recommended action_idx is returned along with the probabilities and the child node.

        Possible upgrades of this approach which will be added later are: early stopping, always selecting winning moves, direchlet noise.
        """
        s = game.canonical_string_rep()
        if s in self.nodes:
            node = self.nodes[s]
        else:
            node = Node(game)
        for _ in range(self.num_sims):
            leaf, hist = self.find_leaf(node)
            if leaf.game_over:
                value = leaf.value
            else:
                if self.use_dummy_nn:
                    child_priors = torch.ones_like(leaf.valid_actions) / len(leaf.valid_actions)
                    value = 0.0
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
                if len(hist) > 0:
                    self.add_node(leaf, hist[-1][0], hist[-1][1])
                else:
                    self.add_node(leaf)

            self.backup(value,leaf.game.current_player, hist)


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
            action_idx = int(np.random.choice(np.arange(len(node.valid_actions)), p=tree_policy.numpy()))
        node_next = self.select_childnode(node,action_idx)
        action = node.valid_actions[action_idx].item()
        return tree_policy, action, node_next.game

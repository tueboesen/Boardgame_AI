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
    def __init__(self, mcts, action, done, reward, game, parent=None):
        # self.game = parent.game
        # self.game = game
        self.game = game
        self.game_local = copy.deepcopy(game)
        self.action = action  # Action used to go to this state (0-5)
        self.board_action = action # The corresponding action on the board (47,566,1570,3004,6043)
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.mcts = mcts

        self.valid_actions = self.game.get_valid_moves()
        self.number_of_actions = len(self.valid_actions)
        self.child_total_value = torch.zeros([self.number_of_actions])  # Q
        self.child_priors = torch.empty([self.number_of_actions])  # P
        self.child_number_visits = torch.zeros([self.number_of_actions])  # N
        # self.valid_actions = obs["action_mask"].astype(np.bool)

        self.reward = reward
        self.done = done
        self.obs = self.game.rep_nn()

    def __repr__(self) -> str:
        return f"Turn={self.game.turn},player={'white' if self.game.whites_turn else 'black'}, visits={self.number_visits}, actions={self.child_number_visits}, child={self.child_priors}"


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
            math.sqrt(self.number_visits)* self.child_priors/ (1 + self.child_number_visits)
        )

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        idx = torch.argmax(child_score)
        return idx

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        # self.action_idx = action_idx
        self.child_priors = child_priors.cpu()

    def get_child(self, action,game=None):
        if action not in self.children:
            if game is None:
                game = self.game
            game_next = copy.deepcopy(game)
            game_action = self.valid_actions[action]
            game_next.perform_action(game_action)

            self.children[action.item()] = Node(
                mcts=self.mcts,
                action=action,
                done=game_next.game_over,
                reward=game_next.reward(),
                game=game_next,
                parent=self,
            )
            #mcts, action, done, reward, board, parent=None
        return self.children[action.item()]

    def backup(self, value,player):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if player == current.game.current_player:
                sign = 1
            else:
                sign = -1
            current.total_value += sign * value
            current = current.parent


class RootParentNode:
    """
    The parent root node, which sits at the top of the recursive tree structure
    """
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
    """
    The Monte Carlo Tree Search algorithm.

    This algorithm works on a tree structure governed by Node.
    When compute_action is

    Note that this algorithm
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
        node = Node(mcts=self, action=None, done=False, reward=0, game=game, parent=RootParentNode(self.game))
        self.node = node
        self.actionhist = []

    def reset(self):
        node = Node(mcts=self, action=None, done=False, reward=0, game=self.game, parent=RootParentNode(self.game))
        self.node = node
        self.actionhist = []

    def update_node(self,actionhist):
        """
        This routine is used when a MCTS-player plays with other players in a game.
        self.compute_action will perform a MCTS and select the most favorable action and advance the node to this state,
        but when other players performs a move we need some way to tell the MCTS-player that the game has advanced to a new state.
        This routine serves that purpose.

        actions should be a list of actions needed to propagate the game from the state saved in self.node, to the desired state.
        """
        node = self.node
        for action in actionhist[len(self.actionhist):]:
            action_idx = (action == node.valid_actions).nonzero().squeeze()
            node = node.get_child(action_idx,node.game_local)
        self.node = node
        self.actionhist = copy.deepcopy(actionhist)
        return
    def compute_action(self):
        """
        When called, this routine performs a number of simulation equal to self.num_sims.
        Each simulation starts from the game state defined in self.node and explores from that state until a leaf node is found (a node that has not previously been explored)
        The leaf node is then evaluated using the model (typically a neural network), and the policy vector and its value is stored in the nodes.
        After the specified number of simulations have been performed, the recommended action_idx is returned along with the probabilities and the child node.
        """
        node = self.node
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
            leaf.backup(value,leaf.game.current_player)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = torch.nn.functional.softmax(tree_policy/self.temperature,dim=0)
        # tree_policy = tree_policy / torch.max(tree_policy)  # to avoid overflows when computing softmax
        # tree_policy = np.power(tree_policy, self.temperature)
        # tree_policy = tree_policy / torch.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action_idx that has the maximum
            # tree policy probability
            action_idx = torch.argmax(tree_policy)
        else:
            # otherwise sample an action_idx according to tree policy probabilities
            action_idx = np.random.choice(np.arange(len(node.valid_actions)), p=tree_policy.numpy())
        if action_idx not in node.children:
            node.get_child(action_idx)
        action = node.valid_actions[action_idx].item()
        self.node = node.children[action_idx]
        self.actionhist.append(action)
        return tree_policy, action, self.node.game

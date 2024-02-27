import graphviz
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
    "num_simulations": 3,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,}

class MCGS:
    """
    The Monte Carlo Graph Search algorithm.

    """

    def __init__(self, model, mcts_param=mcts_config,max_rep=3):
        self.model = model
        self.max_rep = max_rep
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]
        self.visits = {}
        self.priors = {}
        self.child_values = {} # The value is always shown from the current players perspective
        self.child_visits = {}
        self.child_value = {}
        self.valid_actions = {}
        self.child_rep = {}
        self.max_depth = 0



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

    def find_leaf(self,game):
        hist_state = []
        hist_player = []
        hist_action_idx = []
        ii = 0
        while True:
            ii += 1
            s = game.string_rep()
            hist_state.append(s)
            hist_player.append(game.current_player)
            if game.game_over:
                self.visits[s] = 0
                return game, hist_state, hist_player, hist_action_idx
            if not s in self.visits:
                self.visits[s] = 0
                self.child_visits[s] = torch.zeros_like(game.get_valid_moves())
                self.valid_actions[s] = game.get_valid_moves().tolist()
                self.max_depth = max(self.max_depth,ii)
                return game, hist_state, hist_player, hist_action_idx # s is a leaf
            # action_priors = self.priors[s]
            denominator = 1 /  (1 + self.child_visits[s])
            action_probs = self.child_values[s] * denominator + math.sqrt(self.visits[s]) * self.priors[s] * denominator

            counters = torch.ones_like(action_probs)
            for i in range(action_probs.shape[0]):
                # we try the best action according to this, but we might not actually pick it, if it turns out that the action has previously been picked
                action_try_idx = torch.argmax(action_probs/counters).item()
                action_try = self.valid_actions[s][action_try_idx]
                # if (s,action_try) in self.child_rep: # Have we previously checked out this particular action?
                #     s_try = self.child_rep[s,action_try_idx]
                #     game_try = None
                # else:
                game_try = copy.deepcopy(game)
                game_try.perform_action(action_try)
                # game_try = game.getNextState(action_try)
                s_try = game_try.string_rep()
                self.child_rep[s,action_try] = s_try
                count = game.hist.count(s_try) + hist_state.count(s_try) + 1
                if count >= self.max_rep:
                    count = 100000000
                if count == counters[action_try_idx]:
                    break
                else:
                    counters[action_try_idx] = count
            self.child_visits[s][action_try_idx] += 1
            hist_action_idx.append(action_try_idx)
            game = game_try






    def backprop_reward(self,hist_state,hist_player,hist_action_idx,reward):
        player_reward = hist_player.pop()
        s_child = hist_state.pop()
        # action_idx_child = hist_action_idx.pop()
        self.visits[s_child] += 1
        for s,p,idx, in zip(reversed(hist_state),reversed(hist_player),reversed(hist_action_idx)):
            sign = 2 * int(p == player_reward) - 1
            self.child_values[s][idx] += sign * reward
            self.visits[s] += 1
        return

    def draw_node(self,g,s):
        g.node(s,label=f"{s} \n visits:{self.visits[s]} \n value:{self.child_values[s].sum().item():2.4f}. \n child_visits:{self.child_visits[s].tolist()}")
        return s

    def draw_unexplored_node(self,g,s_parent,action_idx):
        name = s_parent+"_"+str(action_idx)
        g.node(name,shape='square')
        return name


    def draw_node_and_children(self,g,s,drawn_nodes,depht=0):
        self.draw_node(g,s)
        drawn_nodes.append(s)
        for visits, action in zip(self.child_visits[s],self.valid_actions[s]):
            if (s,action) in self.child_rep:
                s_child = self.child_rep[s,action]
                # name = self.draw_node(g,s_child)
                if s_child in drawn_nodes:
                    print("already exists")
                    g.edge(s,s_child,color='red')
                else:
                    g = self.draw_node_and_children(g,s_child,drawn_nodes=drawn_nodes,depht=depht+1)
                    g.edge(s,s_child)
            else:
                name = self.draw_unexplored_node(g,s,action)
                g.edge(s,name)
        return g

    def viz(self,game):
        s = game.canonical_string_rep()

        g = graphviz.Digraph(comment='The Round Table')
        # dot.node(s,label = "<1. <br/> 2. <br/>  3. <br/>  4. <br/>  .... <br/>", color="blue", style="dashed")
        g = self.draw_node_and_children(g,s,[])
        g2 = g.unflatten(stagger=8)
        # g2.view()
        g2.format = 'svg'
        g2.render(directory='.')
        # print(g.source)  # doctest: +NORMALIZE_WHITESPACE +NO_EXE

    def compute_action(self,game):
        """
        """
        for _ in range(self.num_sims):
            game_leaf,hist_state,hist_player,hist_action_idx = self.find_leaf(game)
            if game_leaf.game_over:
                reward = game_leaf.reward()
            else:
                action_priors, reward = self.model.predict(game_leaf.rep_nn())
                valid_actions = game_leaf.get_valid_moves()
                action_priors = action_priors[valid_actions]
                action_priors = action_priors / action_priors.sum()
                self.priors[game_leaf.string_rep()] = action_priors.cpu()
                self.child_values[game_leaf.string_rep()] = torch.zeros_like(self.priors[game_leaf.string_rep()])
            self.backprop_reward(hist_state,hist_player,hist_action_idx,reward)
        # print(f"max depth: {self.max_depth}")
        # self.viz(game)
        s = game.canonical_string_rep()
        tree_policy = self.child_visits[s] / self.visits[s]
        tree_policy = torch.nn.functional.softmax(tree_policy/self.temperature,dim=0)
        if self.exploit:
            # if exploit then choose action_idx that has the maximum
            # tree policy probability
            action_idx = torch.argmax(tree_policy)
        else:
            # otherwise sample an action_idx according to tree policy probabilities
            action_idx = np.random.choice(np.arange(len(self.valid_actions[s])), p=tree_policy.numpy())
        action = self.valid_actions[s][action_idx]
        game_next = copy.deepcopy(game)
        game_next.perform_action(action)
        return tree_policy, action, game_next



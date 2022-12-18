import logging
import math

import numpy as np
import torch

from hive.viz import draw_board

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        # self.Qsa = {}  # stores Q values for s,a (as defined in the paper) - Mean value of next state
        self.Wsa = {}
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # Prior probability of selecting action a

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.hist = []

    def Qsa(self,s,a):
        return self.Wsa[(s,a)] / self.Nsa[(s,a)]

    def backup(self):
        # hist = self.hist
        # for s,a in reversed(hist):
        #     self.Ns[s]
        pass

    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            v = self.search(board)
            # self.backup(v)

        s = self.game.stringRepresentation(board)
        action_indices = self.Vs[s]
        counts = [self.Nsa[(s, a.item())] if (s, a.item()) in self.Nsa else 0 for a in self.Vs[s]]


        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs #, action_indices


    def search(self, board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(board)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board)
        if self.Es[s] is not None:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            p, v = self.nnet.predict(board.rep_nn())
            valids = self.game.getValidMoves(board)
            p_val = p[valids]
            self.Ps[s] = p_val / p_val.sum()
            self.Vs[s] = valids
            self.Ns[s] = 0

            return -v

        a = self.pick_action(s)
        next_s = self.game.getNextState(board, a)
        # if board.turn > 1501:
        #     draw_board(board.board_state)
        # if board.turn > 1511:
        #     print("here")

        v = self.search(next_s)
        if board.whites_turn == next_s.whites_turn:
            v = -v
        self.Wsa[(s,a)] += v
        self.Ns[s] += 1
        return -v

    def pick_action(self,s):

        actions_possible = self.Vs[s]
        actions_possible_list = actions_possible.tolist()
        # u_best = -float('inf')
        P = self.Ps[s]
        Ns = self.Ns[s]
        Pns = self.args.cpuct * P * math.sqrt(Ns+EPS)
        idx_W = []
        W = []
        Nsa = []
        idx_nex = []
        for i, a in enumerate(actions_possible_list):
            if (s, a) in self.Wsa:
                idx_W.append(i)
                Nsa.append(self.Nsa[(s,a)])
                W.append(self.Wsa[(s,a)])
            else:
                idx_nex.append(i)
        u2 = torch.empty_like(P)
        Nsa = torch.as_tensor(Nsa,device=P.device)
        W = torch.as_tensor(W,device=P.device)
        u2[idx_W] = W/Nsa + Pns[idx_W] / (1 + Nsa)
        u2[idx_nex] = Pns[idx_nex]
        best_act = actions_possible_list[torch.argmax(u2).item()]

        # pick the action with the highest upper confidence bound
        # for i, a in enumerate(actions_possible):
        #     a = a.item()
        #     if (s, a) in self.Wsa:
        #         u = self.Qsa(s, a) + Pns[i] / (1 + self.Nsa[(s, a)])
        #     else:
        #         u = Pns[i]  # Q = 0 ?
        #
        #     if u > u_best:
        #         u_best = u
        #         best_act = a
        # assert best_act == best_act2
        if (s, best_act) in self.Wsa:
            self.Nsa[(s, best_act)] += 1
        else:
            self.Nsa[(s ,best_act)] = 1
            self.Wsa[(s ,best_act)] = 0
        return best_act

    #
    #
    # def search2(self, board):
    #     """
    #     This function performs one iteration of MCTS. It is recursively called
    #     till a leaf node is found. The action chosen at each node is one that
    #     has the maximum upper confidence bound as in the paper.
    #
    #     Once a leaf node is found, the neural network is called to return an
    #     initial policy P and a value v for the state. This value is propagated
    #     up the search path. In case the leaf node is a terminal state, the
    #     outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
    #     updated.
    #
    #     NOTE: the return values are the negative of the value of the current
    #     state. This is done since v is in [-1,1] and if v is the value of a
    #     state for the current player, then its value is -v for the other player.
    #
    #     Returns:
    #         v: the negative of the value of the current canonicalBoard
    #     """
    #
    #     while True:
    #         s = self.game.stringRepresentation(board)
    #         if s not in self.Es:
    #             self.Es[s] = self.game.getGameEnded(board)
    #         if self.Es[s] is not None:
    #             return -self.Es[s]
    #
    #         if s not in self.Ps:
    #             # leaf node
    #             p, v = self.nnet.predict(board.rep_nn())
    #             valids = self.game.getValidMoves(board)
    #             p_val = p[valids]
    #             self.Ps[s] = p_val / p_val.sum()
    #             self.Vs[s] = valids
    #             self.Ns[s] = 0
    #
    #         for a in valids:
    #
    #
    #             next_s = self.game.step(board, a)
    #
    #         valids = self.Vs[s]
    #         cur_best = -float('inf')
    #         best_act = -1
    #
    #         # pick the action with the highest upper confidence bound
    #         for a in valids:
    #             if (s, a) in self.Qsa:
    #                 u = self.Qsa[(s, a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
    #                         1 + self.Nsa[(s, a)])
    #             else:
    #                 u = self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
    #
    #             if u > cur_best:
    #                 cur_best = u
    #                 best_act = a
    #
    #         a = best_act
    #         next_s = self.game.getNextState(board, a)
    #         # if board.turn > 1501:
    #         #     draw_board(board.board_state)
    #         if board.turn > 1511:
    #             print("here")
    #
    #         v = self.search(next_s)
    #
    #         if (s, a) in self.Qsa:
    #             self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
    #             self.Nsa[(s, a)] += 1
    #
    #         else:
    #             self.Qsa[(s, a)] = v
    #             self.Nsa[(s, a)] = 1
    #
    #         self.Ns[s] += 1
    #         return -v

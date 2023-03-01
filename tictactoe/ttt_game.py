from __future__ import print_function
import sys

import torch

from Templates.game import Game
from tictactoe.ttt_gamelogic import Board

sys.path.append('..')
import numpy as np

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
class TicTacToeGame(Game):
    def __init__(self, n=3):
        self.n = n
        self.board = Board(self.n)
        self.player1_turn = True
        self._previous_player = 0



    @property
    def board_size(self):
        # (a,b) tuple
        return (self.n, self.n)


    @property
    def action_size(self):
        # return number of actions
        return self.n*self.n

    @property
    def number_of_players(self):
        return 2

    @property
    def current_player(self):
        p = 0 if self.player1_turn else 1
        return p

    @property
    def previous_player(self):
        return self._previous_player

    @property
    def opp_player(self):
        p = 1 if self.player1_turn else 0
        return p
    def perform_action(self, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return
        move = (int(action/self.n), action%self.n)
        self.board.execute_move(move, self.current_player)
        self._previous_player = self.current_player
        self.player1_turn = not self.player1_turn
        return

    def get_valid_moves(self):
        # return a fixed size binary vector
        valids = [0]*self.action_size
        legalMoves =  self.board.get_legal_moves()
        if len(legalMoves)==0:
            return torch.Tensor()
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return torch.tensor(valids).nonzero()[:,0]

    @property
    def game_over(self):
        if self.winner is None and self.board.has_legal_moves():
            gg = False
        else:
            gg = True
        return gg

    @property
    def winner(self):
        b = self.board
        if b.is_win(0):
            return 0
        elif b.is_win(1):
            return 1
        else:
            return None

    @property
    def reward(self):
        b = self.board
        if b.is_win(0):
            return [1,-1]
        elif b.is_win(1):
            return [-1,1]
        else:
            return [0,0]

    def nn_rep(self):
        a = torch.tensor(self.board.pieces)
        M1 = a == -1
        M2 = a == 0
        a[M1] = 0
        a[M2] = -1
        a = a.to(dtype=torch.float32)
        # a = torch.tensor(self.board.pieces,dtype=torch.float32)
        if self.current_player == 1:
            a = -a

        return a

    def canonical_string_rep(self):
        # 8x8 numpy array (canonical board)
        return str(self.board.pieces)

    def reset(self):
        self.board = Board(self.n)
        self.player1_turn = True

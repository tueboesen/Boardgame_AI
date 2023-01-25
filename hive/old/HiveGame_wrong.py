import copy

import numpy as np

from Game import Game
from players.Board import Board


def find_matching_coordinates(self, xy):
    indices = []
    for hive in self.hives:
        qr_list = self.qr.tolist()
        idx = [i for i, qr in enumerate(qr_list) if qr[0] == xy[0] and qr[1] == xy[1]]
        indices += idx
    return indices

def get_state(self):
    state = [self.turn,self.whites_turn,self.game_over,self.winner]
    for hive in self.hives:
        state.append(hive.qr)
        state.append(hive.in_play)
        state.append(hive.level)
        state.append(hive.pieces_under)
    state.append(self.get_valid_moves())
    return state

def set_valid_moves(self,move_idx):
    moves = self.hive_player.moves
    moves[:,:,:] = False
    moves1d = moves.view(-1)
    moves1d[move_idx] = True

def set_state(self,state):
    self.turn = state[0]
    self.whites_turn = state[1]
    self.game_over = state[2]
    self.winner = state[3]
    c = 3
    for hive in self.hives:
        hive.qr = state[c+1]
        hive.in_play = state[c+2]
        hive.level = state[c+3]
        hive.pieces_under = state[c+4]
        c += 4
    self.set_valid_moves(state[c+1])


def move_piece(self, idx, q, r):
    hp = self.hive_player
    if piece_symbol(hp.types[idx]) == 'b':
        # First we find any potential pieces on the tile we are about to move to,
        # and lower them 1 level and setting pieces_under beetle to that amount
        move_dst = torch.tensor((q, r))
        move_src = hp.qr[idx].clone()

        indices_all = self.find_matching_coordinates(move_dst.tolist())
        hp.pieces_under[idx] = 0
        for i, hive in enumerate(self.hives):
            indices = indices_all[i]
            for idx in indices:
                if hive.in_play[idx]:
                    hive.level[idx] -= 1
                    hive.pieces_under[idx] += 1
        # Then we move the beetle to that coordinate
        hp.move_piece(idx, q, r)

        # Then we check the coordinates the beetle were at originally and raise the level of all pieces there by one.
        indices_all = self.find_matching_coordinates(move_src.tolist())
        for i, hive in enumerate(self.hives):
            indices = indices_all[i]
            for idx in indices:
                if hive.in_play[idx]:
                    hive.level[idx] += 1
    else:
        hp.move_piece(idx, q, r)


def generate_int_form(self):
    """
    Beetle = 1
    Queen = 2
    Grasshopper = 3
    Spider = 4
    Ant = 5

    current_player is positive numbers
    opp is negative numbers

    level=0 *1
    level=-1 *100
    level=-2 *10000
    level=-3 *1000000
    level=-4 *100000000
    level=-5 *10000000000
    """
    b = torch.zeros(self.board_size, dtype=torch.int64)
    hives = [self.hive_player, self.hive_opp]
    m = -1
    for hive in hives:
        m *= -1
        for (id, in_play, level, qr) in hive:
            if in_play:
                val = id + 1
                b[qr[0], qr[1]] += m * val * 100 ** (level)
    return b


class HiveGame(Game):
    """
    Hive game: https://www.ultraboardgames.com/hive/game-rules.php

    Player 1 = White
    player -1 = Black
    """

    def __init__(self):
        self.board = Board()
        self.board_size = self.board.board_size
        self.action_size = self.board.npieces_per_player * self.board.board_len * self.board.board_len

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = self.board
        return board

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.board_size

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.action_size

    def step(self, action):
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        self.board.perform_action(action)
        obs = self.board.rep_nn()
        reward = self.board.reward()
        done = self.board.game_over
        _ = None
        return obs, reward, done, _

    def set_state(self,state):
        self.board.set_state(state)

    def get_state(self):
        return self.board.get_state()
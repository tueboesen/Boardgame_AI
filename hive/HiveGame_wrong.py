import copy

import numpy as np

from Game import Game
from hive.Board import Board


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
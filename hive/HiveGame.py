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

    def getNextState(self, board, action):
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        board_next = copy.deepcopy(board)
        # try:
        board_next.perform_action(action)
        # except:
        #     board.get_valid_moves()
        #     board.perform_action(action)

        return board_next

    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        moves = board.hive_player().moves
        return moves.view(-1).nonzero()[:,0]
        # return moves.view(-1).numpy()

    def getGameEnded(self, board):
        """
        Input:
            board: current board

        """
        win = board.winner
        if win is None:
            return None
        else:
            quit()
        # elif win == 'White': #White won
        #     return 1
        # elif win == 'Black': #Black won
        #     return -1
        # else:
        #     return 0 #Draw


    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board,pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        We make a canonical board, by assigning a value to each piece. and a multiplier for each level

        """

        # b = copy.deepcopy(board)
        board_s = board.string_rep()
        # board_s = b.rep_str()
        return board_s
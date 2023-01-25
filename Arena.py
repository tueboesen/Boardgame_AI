import logging
import sys

from tqdm import tqdm
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, players, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.players = players
        self.game = game
        self.display = display
        self.board = game.board
    def single_game(self):
        if self.display is not None:
            pygame.display.update()

        while self.game.getGameEnded() is None:
            player = self.players[self.game.current_player]
            action = player.determine_action()
            self.game.getNextState(action)
            if self.display is not None:
                self.display.update_board()
                self.display()

        print("game over!")


    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, self.player1]
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board) is None:
            it += 1
            if verbose:
                assert self.display
                print(f"Turn {it}, Player: {'white' if board.whites_turn else 'black'}")
                self.display(board)
            action = players[board.whites_turn*1](board)

            valids = self.game.getValidMoves(board)

            # if valids[action] == 0:
            #     log.error(f'Action {action} is not valid!')
            #     log.debug(f'valids = {valids}')
            #     assert valids[action] > 0
            board = self.game.getNextState(board, valids[action])
        if verbose:
            assert self.display
            print(f"Game over: Turn {it}, Result {self.game.getGameEnded(board)}")
            self.display(board)
        return self.game.getGameEnded(board)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

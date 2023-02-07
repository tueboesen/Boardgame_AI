import copy
import logging
import sys
from time import sleep
from collections import deque
from tqdm import tqdm
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import numpy as np
log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any number of players can be pitted against each other.
    """

    def __init__(self, players, game, display=None):
        """
        Input:
            player: a list of player objects
            game: Game object
            display: a function that takes game as input and prints it. Is necessary for verbose
                     mode.
        """

        self.players = deque(players)
        self.game = game
        self.display = display

    @property
    def nplayers(self):
        return len(self.players)
    def playGame(self,players=None):
        """
        Plays a single game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
                #TODO THIS is outdated and should be fixed for multiplayer games
        """
        if players is None:
            players = self.players
        game = self.game
        if self.display is not None:
            self.display.update_board(game)
            self.display()
            pygame.display.update()

        actionhist = []
        while not game.game_over:
            player = players[game.current_player]
            action = player.determine_action(game,actionhist)
            game.perform_action(action)
            actionhist.append(action)
            if self.display is not None:
                self.display.update_board(game)
                self.display()
                pygame.display.update()
                # self.display.clock.tick(30)
                sleep(0.01)
        # print("game over!")
        # input("Press any key to continue... ")
        winner = game.winner
        return winner

    def playGames(self, ngames, verbose=False):
        """
            #TODO needs to be updated for multiplayer games
        """
        nplayers = self.nplayers
        wins = np.zeros(nplayers,dtype=np.int64)
        draws = 0
        for i in range(nplayers):
            for _ in tqdm(range(round(ngames/nplayers)), desc="Arena.playGames (1)"):
                winner = self.playGame()
                if winner is not None:
                    wins[(winner+i)%nplayers] += 1
                else:
                    draws += 1
                self.resetGame()
            self.players.rotate(1)
        return wins,draws

    def resetGame(self):
        self.game.reset()
        for player in self.players:
            player.reset()

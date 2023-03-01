import copy

import pygame
import torch
from abc import ABC, abstractmethod

from Templates.game import Game
from Templates.ui import UI
from src.mcts import MCTS


class PlayerTemplate(ABC):
    @abstractmethod
    def determine_action(self,game: Game) -> int:
        """
        Given a game, this function should return an integer corresponding to the action taken
        """
        pass

    def reset(self):
        pass

    def __str__(self):
        return self.__class__.__name__



class HumanPlayer(PlayerTemplate):
    def __init__(self, ui: UI):
        self.ui = ui

    def determine_action(self, game: Game) -> int:
        action = None
        while action is None:
            action = self.ui.get_user_input()
            self.ui.redraw_game()
            pygame.display.update()
            # self.ui.clock.tick(30)
        return action


class RandomPlayer(PlayerTemplate):
    """
    This player will perform a random valid move.
    """
    def __init__(self):
        pass

    def determine_action(self, game: Game) -> int:
        moves = game.get_valid_moves()
        idx = torch.randint(0,moves.shape[0],(1,))
        action = moves[idx].squeeze()
        return action


class MCTSNNPlayer(PlayerTemplate):
    """
    This is a Monte Carlo Tree Search Neural Network player.
    In order for this player to work it needs a neural trained neural network that can somewhat accurately predict the
    action probability as well as the expected outcome.
    """
    def __init__(self,mcts: MCTS,description=None):
        self.mcts = mcts
        self.description = description

    def determine_action(self,game: Game) -> int:
        move_prob, action, node_next = self.mcts.compute_action(game)
        return action

    def reset(self):
        self.mcts.reset()

    # def __repr__(self):
    #     cls = self.__class__.__name__
    #     return f'{cls} (description={self.description!r}, mcts={self.mcts!r})'

    def __str__(self):
        if self.description is None:
            return self.__class__.__name__
        else:
            return self.description

HUMAN = {'fnc': HumanPlayer,
         'req': [("viz", UI)],
         }
RANDOM = {'fnc': RandomPlayer,
         'req': [],
         }
MCTSNN = {'fnc': MCTSNNPlayer,
         'req': [("mcts",MCTS)],
         }

PLAYMODES = [RANDOM,MCTSNN] # We skip Human player since we can't really do unit testing on those
import copy

import pygame
import torch
from abc import ABC, abstractmethod

from Templates.game import Game


class PlayerTemplate(ABC):
    @abstractmethod
    def determine_action(self,game: Game) -> int:
        """
        Given a game, this function should return an integer corresponding to the action taken
        """
        pass

    def reset(self):
        pass

class HumanPlayer(PlayerTemplate):
    def __init__(self,display):
        self.display = display

    def determine_action(self, game: Game) -> int:
        action = None
        while action is None:
            self.display.get_mouse_hover()
            action = self.display.handle_mouse_click()
            self.display.redraw_game()
            pygame.display.update()
            self.display.clock.tick(30)
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
    def __init__(self,mcts,description=None):
        self.mcts = mcts
        self.description = description

    def determine_action(self,game: Game) -> int:
        move_prob, action, node_next = self.mcts.compute_action(game)
        return action

    def reset(self):
        self.mcts.reset()

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls} (description={self.description!r}, mcts={self.mcts!r})'

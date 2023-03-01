from abc import ABC, abstractmethod

class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    @property
    @abstractmethod
    def action_size(self):
        """
        Returns:
            number of all possible actions
        """
        pass

    @property
    @abstractmethod
    def number_of_players(self):
        """
        Returns:
            Number of players for the game
        """

    @property
    @abstractmethod
    def current_player(self):
        """
        Returns the current player 0=player1, 1=player2, 2=player3, ect...
        """

    @abstractmethod
    def perform_action(self, action_idx):
        """
        Advances the game state by performing the action given by action_idx, note that the game state should be internal.
        """
        pass

    @abstractmethod
    def get_valid_moves(self):
        """
        Returns:
            validMoves: a vector which contains the indices for all valid actions
        """
        pass

    @property
    @abstractmethod
    def game_over(self):
        """
        Returns:
            r: returns True if game is over, false otherwise.

        """
        pass

    @property
    @abstractmethod
    def reward(self):
        """
        Returns:
            A vector with the reward for all players
        """
        pass

    @abstractmethod
    def canonical_string_rep(self):
        """
        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    @abstractmethod
    def nn_rep(self):
        """
        Returns:
            A representation of the board used as input for neural networks.
        """
        pass

    @abstractmethod
    def reset(self):
        pass
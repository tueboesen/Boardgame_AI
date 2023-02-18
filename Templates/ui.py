from abc import ABC, abstractmethod

class UI(ABC):
    @abstractmethod
    def __init__(self,game):
        pass


    @abstractmethod
    def redraw_game(self):
        """
        A method that draws the board based on the information currently saved in UI
        """
        pass

    @abstractmethod
    def sync_ui_to_game(self, game):
        """
        A method that takes the current game state and updates UI with the relevant information
        """
        pass


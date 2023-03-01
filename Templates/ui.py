from abc import ABC, abstractmethod

from Templates.viz import Vizualization


class UI(Vizualization):
    @abstractmethod
    def __init__(self,game):
        pass


    @abstractmethod
    def get_user_input(self):
        """
        A method that gets the users input either in terms of mouse position/clicks or in terms of keyboard input
        """
        pass


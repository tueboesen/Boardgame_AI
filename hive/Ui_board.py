from math import cos, sin, pi, radians

import numpy as np
import pygame
import torch
from pygame import gfxdraw
from pygame import time


class Hexes:
    """
    This contains all the hexes that makes up the board the pieces move around on
    """
    def __init__(self, screen, offset, hex_radius, color_default, color_hover, color_selected, color_highlight):
        self.screen = screen
        self.offset = offset
        self.hex_radius = hex_radius
        self.color_default = color_default
        self.color_hover = color_hover
        self.black = (40,40,40)
        self.color_selected = color_selected
        self.color_highlight = color_highlight
        self.hexes = []
        self.hex_selected = None
        self.hex_hovered = None
        self.n = 6
        self.update_board = True

    def __len__(self):
        return len(self.hexes)

    def __iter__(self):
        yield from self.hexes

    def __getitem__(self, item):
        return self.hexes[item]

    def create_hex(self, x, y, color=None, symbol=None):
        color = self.color_default if color is None else color
        hex = Hexagon(x, y, color, self.screen, self.offset, self.hex_radius, symbol)
        self.hexes.append(hex)

    def draw_hex(self, idx: int):
        hex = self.hexes[idx]
        hex.draw()

    def draw_hexes(self):
        for i in range(self.__len__()):
            self.draw_hex(i)
        self.update_board = False

    def draw_highlight(self, idx: int):
        self.hexes[idx].draw_highlight()

    def draw_hover(self, idx: int):
        self.hexes[idx].draw_hover()


    def draw_highlights(self):
        for i in range(self.__len__()):
            self.draw_highlight(i)
        # self.update_board = False


    def move_hex(self, idx: int, x: float, y: float):
        hex = self.hexes[idx]
        hex.move(x, y)

    # def highlight_hex(self, indices: list):
    #     for idx in indices:
    #         self.hexes[idx].color_outer_edge = self.color_highlight

    def select_hex(self, idx: int):
        if idx == self.hex_selected: # Unselect hex
            self.reset_color_hexes()
            # self.hexes[idx].color_fill = self.color_default
            self.hex_selected = None
            self.update_board = True
        elif self.hex_selected is None: # Select hex
            self.hexes[idx].color_fill = self.color_selected
            self.hex_selected = idx
            self.update_board = True

    def hover_hex(self, idx: int):
        if idx == self.hex_hovered: # Nothing has changed
            return
        if self.hex_hovered is not None: # Remove old hovered hex
            self.hexes[self.hex_hovered].color_inner_edge = self.black
            self.hex_hovered = None
        if idx is not None: # Set new hovered hex
            self.hexes[idx].color_inner_edge = self.color_hover
            self.hex_hovered = idx
        self.update_board = True

    def reset_color_hex(self, idx: int):
        self.hexes[idx].reset_color()

    def reset_color_hexes(self):
        for i in range(self.__len__()):
            self.reset_color_hex(i)


class Hexagon(Hexes):
    """
    This is a single hexagon on the board
    """
    def __init__(self, x, y, color, screen, offset, hex_radius, symbol=None, color_hover=(0, 121, 251), color_text=(40, 40, 40), color_text_bg=(0, 121, 251)):
        self.n = 6
        self.x = x
        self.y = y
        self.offset = offset
        self.hex_radius = hex_radius
        self.color_fill = color
        self._color_fill = color
        self.color_inner_edge = (40, 40, 40)
        self._color_inner_edge = (40, 40, 40)
        self.color_outer_edge = (0,255,0) #(40, 40, 40)
        self._color_outer_edge = (40, 40, 40)
        self.color_hover = color_hover
        self.symbol = symbol
        self.screen = screen

        self.node_font = pygame.font.SysFont("Sans", 18)
        self.color_text = color_text
        self.color_text_bg = color_text_bg

        self.move(x, y)

    def __contains__(self, key):
        return key in self.rect

    def calculate_xy_outer(self, x, y):
        self.xy_outer = [(x + (self.hex_radius + self.offset) * cos(radians(90) + 2 * pi * _ / self.n),
                          y + (self.hex_radius + self.offset) * sin(radians(90) + 2 * pi * _ / self.n))
                         for _ in range(self.n)]

    def calculate_xy_inner(self, x, y):
        self.xy_inner = [(x + (self.hex_radius) * cos(radians(90) + 2 * pi * _ / self.n),
                          y + (self.hex_radius) * sin(radians(90) + 2 * pi * _ / self.n))
                         for _ in range(self.n)]

    def calculate_xy_rect(self, x, y):
        self.xy_rect = (x - self.hex_radius + 3*self.offset, y - (self.hex_radius / 2),
                        (self.hex_radius * 2) - (6 * self.offset), self.hex_radius)


    def move(self, x, y):
        self.x = x
        self.y = y
        self.calculate_xy_outer(x, y)
        self.calculate_xy_inner(x, y)
        self.calculate_xy_rect(x, y)
        self.rect = pygame.draw.rect(self.screen, self.color_fill, pygame.Rect(self.xy_rect))

    def reset_color(self):
        self.color_fill = self._color_fill
        self.color_inner_edge = self._color_inner_edge
        self.color_outer_edge = self._color_outer_edge

    def draw(self):
        # gfxdraw.aapolygon(self.screen, self.xy_outer, self.color_edge) # Outer hexagon edge
        gfxdraw.filled_polygon(self.screen, self.xy_inner, self.color_fill) # Inner filled hexagon
        gfxdraw.aapolygon(self.screen, self.xy_inner, self.color_inner_edge) # Inner hexagon edge

        if self.symbol is not None:
            text = self.node_font.render(self.symbol, True, self.color_text) #, self.color_text_bg
            text_rect = text.get_rect()
            text_rect.center = (self.x, self.y)
            self.screen.blit(text, text_rect)

    def draw_highlight(self):
        gfxdraw.aapolygon(self.screen, self.xy_outer, self.color_outer_edge) # Outer hexagon edge

    def draw_hover(self):
        gfxdraw.aapolygon(self.screen, self.xy_outer, self.color_hover) # Outer hexagon edge

    def pos(self):
        return (self.x, self.y)


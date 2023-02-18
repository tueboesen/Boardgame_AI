import pygame
from pygame import gfxdraw, time

from Templates.ui import UI


class TicTacToeUI(UI):
    def __init__(self,game):
        pygame.init()
        pygame.display.set_caption("TicTacToe")
        self.display_len = 1000
        self.screen = pygame.display.set_mode((self.display_len,self.display_len))
        self.draw_background()
        # self.directions = game.board.__directions
        self.n = 3
        self.pieces = game.board.pieces
        self.clock = time.Clock()



    def redraw_game(self):
        self.screen.fill((0,0,0))
        self.draw_background()

        for y in range(self.n):
            for x in range(self.n):
                piece = self.pieces[x][y]
                if piece != -1:
                    self.draw_piece(x,y,piece)

    def sync_ui_to_game(self, game):
        self.pieces = game.board.pieces



    def draw_background(self):
        # gfxdraw.filled_polygon(self.screen, self.xy_inner, self.color_fill)
        gfxdraw.hline(self.screen,0,self.display_len,self.display_len//3,(100,100,100))
        gfxdraw.hline(self.screen,0,self.display_len,self.display_len//3*2,(100,100,100))
        gfxdraw.vline(self.screen,self.display_len//3,0,self.display_len,(100,100,100))
        gfxdraw.vline(self.screen,self.display_len//3*2,0,self.display_len,(100,100,100))
        # pygame.display.update()

    def draw_piece(self,x,y,player):
        # gfxdraw.hline(self.screen, 0, self.display_len, self.display_len // 3, (100, 100, 100))
        offset = self.display_len/6
        x_center = self.display_len/3 * x + offset
        y_center = self.display_len/3 * y + offset
        r = (self.display_len/6)*0.9
        if player == 0:
            gfxdraw.circle(self.screen,int(x_center),int(y_center),int(r),(100,100,100))
        else:
            gfxdraw.line(self.screen,int(x_center-r),int(y_center-r),int(x_center+r),int(y_center+r),(100,100,100))
            gfxdraw.line(self.screen,int(x_center-r),int(y_center+r),int(x_center+r),int(y_center-r),(100,100,100))



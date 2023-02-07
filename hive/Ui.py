from math import pi

import pygame
import torch
from pygame import gfxdraw
from pygame import time

from hive.HiveGameLogic_utils import piece_symbol, axial_to_cube_ext
from hive.UiConstants import *
from hive.Ui_board import Hexes


class UI:
    """
    This is the UI for the game hive.

    The overall idea is that you attach this to the game.
    Once the UI is instantiated it will draw an initial board for the game, and save the positions of the pieces for faster drawing/updates.
    If the position of any piece changes, the ui needs to recalculate positions for pieces by calling self.update_board()
    A __call__ routine which will redraw the board and update hover positions, but will not update piece positions unless self.update_board() has been called.


    """
    def __init__(self, game):
        pygame.init()
        pygame.display.set_caption("Hive")
        self.n = 6
        # self.board = game.board
        self.board_size = game.board_size[0]
        # self.hives = game.board.hives
        # assert 1 < self.board_size <= 26
        # self.hive_white = game.board.hive_white
        # self.hive_black = game.board.hive_black
        self.clock = time.Clock()
        self.hex_radius = HEX_RADIUS
        self.hex_ios = HEX_INNER_OUTER_SPACING
        self.x_offset = 4 * HEX_RADIUS + 60
        self.y_offset = 60 + HEX_RADIUS
        # self.text_offset = 45
        self.screen = pygame.display.set_mode(
            (1.5 * self.x_offset + (2 * self.hex_radius) * self.board_size + self.hex_radius * self.board_size,
             round(self.y_offset + (1.75 * self.hex_radius) * self.board_size)))
        self.white = (255, 255, 255)
        self.color_highlight = (0, 255, 0)
        self.color_hover = (0, 121, 251)
        self.black = (40, 40, 40)
        self.gray = (70, 70, 70)
        self.brown = (164, 116, 73)
        self.gold = (255, 215, 0)
        self.board_color = self.brown
        self.piece_hovered = None
        self.piece_selected = None
        self.board_hovered = None
        self.dq = 0
        self.dr = 0
        self.summary = ''
        self.screen.fill(self.gray)
        self.fonts = pygame.font.SysFont("Sans", 20)
        self.hexes_board = Hexes(self.screen, self.hex_ios, self.hex_radius, self.brown, self.color_hover, None, self.color_highlight)
        self.hexes_hives = []
        for hive in game.hives:
            self.hexes_hives.append(HexCoordinates(hive, self.screen))
        self.create_board()
        self.update_board(game)
        self.__call__()


    def __call__(self):
        self.redraw_board()
    def calculate_xy_outer(self, xy):
        n = torch.arange(6)
        x = xy[:,0,None] + (self.hex_radius + self.hex_ios) * torch.cos(pi / 2 + 2 * pi * n / 6)[None,:]
        y = xy[:,1,None] + (self.hex_radius + self.hex_ios) * torch.sin(pi / 2 + 2 * pi * n / 6)[None,:]
        xy_outer = torch.stack((x,y),dim=-1)
        return xy_outer

    def calculate_xy_inner(self, xy):
        a = torch.arange(6)
        x = xy[:,0,None] + (self.hex_radius) * torch.cos(pi / 2 + 2 * pi * a / 6)[None,:]
        y = xy[:,1,None] + (self.hex_radius) * torch.sin(pi / 2 + 2 * pi * a / 6)[None,:]
        xy_inner = torch.stack((x,y),dim=-1)
        return xy_inner

    def calculate_xy_rect(self, xy):
        x = xy[:,0] - self.hex_radius + 3 * self.hex_ios
        y = xy[:,1] - (self.hex_radius / 2)
        dx = torch.ones_like(x) * ((self.hex_radius * 2) - (6 * self.hex_ios))
        dy = torch.ones_like(x) * self.hex_radius
        xy_rect = torch.stack((x,y,dx,dy),dim=-1)
        return xy_rect

    def get_coordinates(self, row, column):
        x = self.x_offset + (2 * self.hex_radius) * column + self.hex_radius * row
        y = self.y_offset + (1.75 * self.hex_radius) * row
        xy = torch.stack((x,y),dim=-1)
        return xy



    def convert_idx_to_row_col(self, idx):
        row = idx // self.board_size
        col = idx % self.board_size
        return row, col

    def convert_bitboard_to_indices(self, bitboard: torch.Tensor) -> list:
        bitboard = bitboard[1:-1, 1:-1]
        indices = torch.flatten(bitboard).nonzero()
        indices = indices.squeeze().tolist()
        if type(indices) != list:
            indices = [indices]
        return indices

    def create_board(self):
        for row in range(self.board_size):
            for column in range(self.board_size):
                xy = self.get_coordinates(torch.tensor(row), torch.tensor(column))
                self.hexes_board.create_hex(xy[0].item(), xy[1].item())

    def find_shift_from_edges_of_board(self,game):
        qmin = torch.tensor(999)
        qmax = -torch.tensor(999)
        rmin = torch.tensor(999)
        rmax = -torch.tensor(999)
        for hive in game.hives:
            qr_viz = game.rep_viz(hive)
            m = hive.in_play
            q = qr_viz[m, 0]
            r = qr_viz[m, 1]
            if len(q)>0:
                qmin = min(torch.min(q),qmin)
                qmax = max(torch.max(q),qmax)
                rmin = min(torch.min(r),rmin)
                rmax = max(torch.max(r),rmax)
        dq = min(self.board_size-2-qmax,max(1 - qmin,0)) #<0
        dr = min(self.board_size-2-rmax,max(1 - rmin,0)) #<0
        self.dqr = torch.tensor([dq,dr])
        return


    def update_board(self,game):
        self.summary = game.summary
        self.piece_selected = None
        self.find_shift_from_edges_of_board(game)
        for hive, hex_coords in zip(game.hives, self.hexes_hives):
            n = len(hive.qr)
            color_white = hive.white
            color_base = PLAYER1 if color_white else PLAYER2
            qr_viz = game.rep_viz(hive)
            qr_viz[hive.in_play] = qr_viz[hive.in_play] + self.dqr
            xy = self.get_coordinates(qr_viz[:,0],qr_viz[:,1])
            xy_inner = self.calculate_xy_inner(xy)
            xy_outer = self.calculate_xy_outer(xy)
            xy_rect = self.calculate_xy_rect(xy)
            viz = hive.level == 0
            color_edge = torch.ones(n,1) * torch.tensor(color_base)
            hex_coords.update(torch.arange(n), viz, xy, xy_inner, xy_outer, xy_rect, color_edge, color_edge)
        moves = game.hive_player.moves.nonzero()
        qr_moves = moves[:,1:]
        qr_viz_moves = game.transform.inverse(qr_moves) + self.dqr
        self.moves = torch.cat((moves[:,:1],qr_viz_moves),dim=1)


    def redraw_board(self):
        self.screen.fill(self.gray)
        self.draw_text_summary()
        self.hexes_board.draw_hexes()
        for hive_hexes in self.hexes_hives:
            hive_hexes.draw_hexes()
        self.draw_selected()
        self.draw_hover_effect()

    def draw_selected(self):
        if self.piece_selected is not None:
            color = PLAYER1_SEL if self.piece_selected[0] == 0 else PLAYER2_SEL
            self.hexes_hives[self.piece_selected[0]].draw_hex(self.piece_selected[1], color_inner=color)
            M = self.moves[:,0] == self.piece_selected[1]
            possible_moves = self.moves[M,1:]
            # possible_moves = self.hives[self.piece_selected[0]].moves[self.piece_selected[1]].nonzero()
            for move in possible_moves:
                i = move[0].item()*self.board_size + move[1]
                # self.hexes_board.highlight_hex([i])
                self.hexes_board.draw_highlight(i)

    def draw_hover_effect(self):
        if self.board_hovered is not None:
            self.hexes_board.draw_hover(self.board_hovered)
        if self.piece_hovered is not None:
            hive_hexes = self.hexes_hives[self.piece_hovered[0]]
            hive_hexes.draw_outline(self.piece_hovered[1],color=HOVER)
    def draw_text_summary(self):
        node_font = pygame.font.SysFont("Sans", 18)
        foreground = self.black
        text = node_font.render(self.summary, True, foreground, self.white)
        text_rect = text.get_rect()
        text_rect.center = (self.screen.get_width() / 2, 20)
        self.screen.blit(text, text_rect)

    def get_coordinates_starting_pieces(self, left: bool, index: int):
        col = -2 if left else self.board_size + 1
        return self.get_coordinates(2 * index, col)

    def get_true_coordinates(self, node: int):
        return int(node / self.board_size), node % self.board_size

    def get_mouse_hover(self):
        """
        Determines which hex and piece the mouse is hovering over
        :return:
        """
        mouse_pos = pygame.mouse.get_pos()
        piece_index = None
        for j,hive_hexes in enumerate(self.hexes_hives):
            for i in range(hive_hexes.n):
                rect = hive_hexes.rect[i]
                if rect is not None and rect.collidepoint(mouse_pos):
                    piece_index = (j,i)
                    break
        self.piece_hovered = piece_index
        board_index = None
        for i, hex in enumerate(self.hexes_board):
            if hex.rect.collidepoint(mouse_pos):
                board_index = i
                break
        self.board_hovered = board_index

    def handle_mouse_click(self):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if self.piece_hovered is not None or self.board_hovered is not None:
                    action = self.click_element()
        return action

    def click_element(self):
        if self.board_hovered is not None and self.piece_selected is not None:  # Attempt to move hex_sel to hex_idx
            row, col = self.convert_idx_to_row_col(self.board_hovered)
            M = self.moves[:,0] == self.piece_selected[1]
            qr = torch.tensor([[row, col]])
            if (self.moves[M,1:] == qr).all(dim=1).any():
            # hp = self.board.hive_player
            # if hp.moves[self.piece_selected[1],row,col]: # Move allowed?
                qr_canon = self.board.transform.forward(qr-self.dqr)
                a = torch.arange(11 * 24 * 24)
                b = a.view(11, 24, 24)
                action_idx = b[self.piece_selected[1],qr_canon[0,0],qr_canon[0,1]]
                return action_idx
        if self.piece_hovered is not None and self.piece_hovered == self.piece_selected:  # unselect piece
            self.piece_selected = None
        elif self.piece_hovered is not None and self.piece_selected is None:
            if (self.board.whites_turn and self.piece_hovered[0] == 0) or (not self.board.whites_turn and self.piece_hovered[0] == 1):
                self.piece_selected = self.piece_hovered
        return



class HexCoordinates:
    def __init__(self,hive,screen):
        self.hive = hive
        n = len(hive.qr)
        self.screen = screen
        self.n = n
        self._visible = torch.empty(n,dtype=torch.bool)
        self._xy = torch.empty((n,2))
        self._xy_inner = torch.empty((n,6,2))
        self._xy_outer = torch.empty((n,6,2))
        self._xy_rect = torch.empty((n,4))
        self._color_inner = torch.empty((n,3))
        self._color_edge = torch.empty((n,3))
        self.rect = [None]*n
        self._i = 0
    # def add(self, viz,xy,xy_inner,xy_rect,color_inner,color_edge):
    #     self._visible.append(viz)
    #     self._xy.append(xy)
    #     self._xy_inner.append(xy_inner)
    #     self._xy_rect.append(xy_rect)
    #     self._color_inner.append(color_inner)
    #     self._color_edge.append(color_edge)

    def update(self,i,viz,xy,xy_inner,xy_outer,xy_rect,color_inner,color_edge):
        self._visible[i] = viz
        self._xy[i] = xy
        self._xy_inner[i] = xy_inner
        self._xy_outer[i] = xy_outer
        self._xy_rect[i] = xy_rect
        self._color_inner[i] = color_inner
        self._color_edge[i] = color_edge

    def __len__(self):
        return self.n

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self):
            result = (self._visible[self._i], self._xy[self._i], self._xy_inner[self._i], self._xy_rect[self._i], self._color_inner[self._i], self._color_edge[self._i])
            self._i += 1
            return result
        else:
            raise StopIteration

    def draw_hexes(self):
        for i in range(self.n):
            if self._visible[i]:
                self.draw_and_save_rect(i)
                self.draw_hex(i)
            else:
                self.rect[i] = None

    def draw_hex(self,i,color_inner=None):
        if color_inner is None:
            color_inner = self._color_inner[i].numpy()
        gfxdraw.filled_polygon(self.screen, self._xy_inner[i].numpy(), color_inner)
        gfxdraw.aapolygon(self.screen, self._xy_inner[i].numpy(), color_inner) # Inner hexagon edge
        sym = piece_symbol(self.hive.types[i])
        text = pygame.font.SysFont("Sans", 18).render(f"{sym if self.hive.pieces_under[i] == 0 else sym + '+' + str(self.hive.pieces_under[i].item())}", True,TEXT)  # , self.ui_color_text_bg
        text_rect = text.get_rect()
        text_rect.center = (self._xy[i][0].item(), self._xy[i][1].item())
        self.screen.blit(text, text_rect)

    def draw_outline(self,i,color=None):
        if color is None:
            color = self._color_edge[i]
        gfxdraw.aapolygon(self.screen, self._xy_outer[i].numpy(), color)

    def draw_and_save_rect(self,i):
        # c = tuple(int(self._color_inner[i].numpy()))
        # r = tuple(int(self._xy_rect[i].numpy()))
        self.rect[i] = pygame.draw.rect(self.screen, tuple(self._color_inner[i].to(dtype=torch.int).numpy()), pygame.Rect(tuple(self._xy_rect[i].to(dtype=torch.int).numpy())))



class TransformMatrix:
    def __init__(self):
        """
        A transformation matrix that converts homogenous hexagonal coordinates in cube format between the canonical representation and the vizualization representation.
        """
        self.A = torch.eye(4)
        self.extra_translation = torch.eye(4)

    def forward(self,qr,A=None):
        qrst = axial_to_cube_ext(qr)
        if A is None:
            A = self.A
        qrst_trans = qrst.to(dtype=torch.float) @ A.T
        qr_trans = qrst_trans[:,:2]
        qr_trans_int = qr_trans.round(decimals=0).long()
        return qr_trans_int
    def inverse(self,qr,A=None):
        qrst = axial_to_cube_ext(qr)
        if A is None:
            A = self.A
        qrst_trans = qrst.to(dtype=torch.float) @ torch.linalg.inv(A.T)
        qr_trans = qrst_trans[:,:2]
        qr_trans_int = qr_trans.round(decimals=0).long()
        return qr_trans_int


    def reset(self):
        self.A = torch.eye(4,device=self.A.device)
    def update(self,A):
        self.A = A @ self.A
        return

    def test(self):
        qr = torch.randint(-10,20,(10,2))
        qr_forward = self.forward(qr)
        qr2 = self.inverse(qr_forward)
        assert qr == qr2

        q = torch.arange(-10,20)
        r = torch.arange(-20,10)
        qr = torch.stack((q,r),dim=1)
        qr_forward = self.forward(qr)

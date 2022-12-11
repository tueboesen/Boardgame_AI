import copy
import typing

import numpy as np
import torch

from hive.Hive import Hive
from hive.HiveGameLogic_utils import piece_id, piece_name, piece_symbol, generate_board, bitmap_get_neighbors, remove_chokepoint_moves, remove_chokepoint_move_indices, generate_conv_board, PIECES_PER_PLAYER, BoolTensor, generate_graph, find_moveable_nodes, \
    DIRECTIONS
from hive.Piece import calculate_moves


class Board():
    # list of 6 neighbouring directions for a hexagonal board using axial coordinates.
    def __init__(self):
        self.hive_white = Hive(white=True)
        self.hive_black = Hive(white=False)
        self.hives = [self.hive_white, self.hive_black]
        self.turn = 1
        self.whites_turn = True
        self.board_len = len(self.hive_white) + len(self.hive_black) + 2
        self.board_size = (self.board_len, self.board_len)
        self.winner = None
        self.game_over = False
        self.npieces_per_player = len(self.hive_white)
        self.get_valid_moves()

    def __repr__(self) -> str:
        return f"Turn={self.turn},player={'white' if self.whites_turn else 'black'}"

    # def __str__(self) -> str:
    #     return f"Turn={self.turn},player={'white' if self.whites_turn else 'black'}"

    def hive_player(self):
        return self.hive_white if self.whites_turn else self.hive_black

    def hive_opp(self):
        return self.hive_black if self.whites_turn else self.hive_white

    def rep_nn(self):
        """
        This should always be from the perspective of the player
        """
        bbs = []
        hives = [self.hive_player(),self.hive_opp()]
        for hive in hives:
            for i, (id, in_play, level, qr) in enumerate(hive):
                if in_play:
                    bb = generate_conv_board(self.board_len, qr)
                    bb /= -level + 1
                else:
                    bb = generate_conv_board(self.board_len)
                bbs.append(bb)
        bbs = torch.stack(bbs)
        return bbs

    def rep_str(self):
        self.fixate_board()
        board_int = self.generate_int_form()
        board_str = np.array2string(board_int.view(-1).numpy(), max_line_width=9999)[1:-1]
        return board_str

    def get_valid_moves(self):
        hive_player = self.hive_player()
        hive_opp = self.hive_opp()
        hive_player.moves[:, :, :] = False

        board_state_player = generate_board(self.board_len, bit=False)
        board_state_opp = generate_board(self.board_len, bit=False)
        bit_state_player = board_state_player.bool()
        bit_state_opp = board_state_opp.bool()
        # First we generate a list of pieces already on the board
        for hive,board,bit in zip([hive_player,hive_opp],[board_state_player,board_state_opp],[bit_state_player,bit_state_opp]):
            for (id, in_play, level, qr) in hive:
                if in_play:
                    board[qr[0],qr[1]] += 1
                    if level == 0:
                        bit[qr[0],qr[1]] = True

        self.board_state = board_state_player + board_state_opp
        self.bit_state = bit_state_player | bit_state_opp  # Note that bit_state is not just bool() of board_state since bitstate does not count piece not at level 0.

        if hive_player.check_played_all():  # Generate list of spawn locations
            spawn_locations = None
        else:
            if hive_player.played_piece():
                bit_player_nn = bitmap_get_neighbors(bit_state_player)
                bit_opp_nn = bitmap_get_neighbors(bit_state_opp)
                spawn_locations = bit_player_nn & (~ (bit_opp_nn | bit_state_opp)) & ~ bit_state_player
            else:
                if hive_opp.played_piece():
                    spawn_locations = torch.roll(bit_state_opp, 1, dims=0) #No reason to give all 6, when they are all the same up to rotation
                else:
                    spawn_locations = generate_board(self.board_len,[(0,0)],bit=True) #No reason to give the whole board

        # Then we go through each piece and see where it can move to
        if hive_player.played_queen():
            g, nodes = generate_graph(self.board_state)
            moveable_node_indices = find_moveable_nodes(g)
            moveable_positions = nodes[moveable_node_indices].tolist()
            for i, (id,in_play,level,qr) in enumerate(hive_player):
                if in_play==False:
                    hive_player.moves[i,:,:] = spawn_locations
                elif level == 0 and qr.tolist() in moveable_positions:
                    state = self.board_state if id == 1 else self.bit_state
                    hive_player.moves[i,:,:] = calculate_moves(id,qr,state)

            if not hive_player.can_move():
                self.next_player()
                self.get_valid_moves()
        elif (hive_player.in_play == True).sum() < 3:
            for i, (id,in_play,level,qr) in enumerate(hive_player):
                if in_play==False:
                    hive_player.moves[i,:,:] = spawn_locations
        else:
            hive_player.moves[0,:,:] = spawn_locations
        return hive_player.moves

    def next_player(self):
        """
        Advances the turn to the next player and checks whether the game is over
        :return:
        """
        self.fixate_board()
        self.check_winners()
        self.whites_turn = not self.whites_turn
        if self.whites_turn:
            self.turn += 1
        self.get_valid_moves()

    def check_winners(self):
        for hive in self.hives:
            surrounded = True
            if hive.in_play[0]:
                qr = hive.qr[0]
                for direction in DIRECTIONS:
                    qr_nn = qr+torch.tensor(direction)
                    if self.check_if_coordinate_filled(qr_nn,self.hive_white) or self.check_if_coordinate_filled(qr_nn,self.hive_black):
                        pass
                    else:
                        surrounded = False
                        break
                if surrounded:
                    hive.lost = True
                    self.game_over = True
        if self.game_over:
            if self.hive_white.lost and self.hive_black.lost:
                self.winner = 'Draw'
            else:
                self.winner = 'Black' if self.hive_white.lost else 'White'
            return



    def check_if_coordinate_filled(self,qr,hive):
        m1 = hive.qr[:,0] == qr[0]
        m2 = hive.qr[:,1] == qr[1]
        m = m1 & m2
        return m.any()





    def shift_pieces_from_edges(self):
        """
        """
        if not self.hive_white.played_piece():
            return

        hives = [self.hive_white, self.hive_black]
        qmin = torch.tensor(999)
        qmax = -torch.tensor(999)
        rmin = torch.tensor(999)
        rmax = -torch.tensor(999)
        for hive in hives:
            m = hive.in_play
            q = hive.qr[m, 0]
            r = hive.qr[m, 1]
            if len(q)>0:
                qmin = min(torch.min(q),qmin)
                qmax = max(torch.max(q),qmax)
                rmin = min(torch.min(r),rmin)
                rmax = max(torch.max(r),rmax)

        dq = min(self.board_len-1-qmax,max(1 - qmin,0)) #<0
        dr = min(self.board_len-1-rmax,max(1 - rmin,0)) #<0
        if dq != 0 or dr != 0:
            hives = [self.hive_white, self.hive_black]
            for hive in hives:
                m = hive.in_play
                hive.qr[m,0] += dq
                hive.qr[m,1] += dr
        return

        

    def fixate_board(self):
        """
        This routine performs modulo 60 degree rotations
        and integer translations to the board such that the first white piece is on (0,0)
        and the first black piece is at (x,y),x>0,y>0 (the first quadrant)
        if multiple rotations ends in the first quadrant,
        we take the counterclockwise rotation that ends there but didn't start there.
        Furthermore it mirrors the board along the x-axis such that the first piece in play not residing on the x-axis has a positive y value.

        This does not completely fixate the board since we do not account for piece permutation, but in practice we should at most have 2 identical boards since we are using the first pieces
        of which there only exist 2 of each,
        """
        hw = self.hive_white
        hb = self.hive_black
        mw = hw.in_play
        mb = hb.in_play
        print(f"White: {hw.qr},{hw.in_play*1}")
        print(f"Black: {hb.qr},{hb.in_play*1}")

        hw_idx = hw.in_play.nonzero()
        hb_idx = hb.in_play.nonzero()

        # We need to uniquely fixate the board such that any identical board states ends up with the same representation
        # Note that this representation is done in such a way that the view is always from the player
        # Meaning that the boardstate where black has an ant and white has a spider and it is blacks turn to play
        # should be identical to the state where black has a spider and white has an ant, and it is whites turn to play

        # First translation
        # 1) players queen
        # 2) Opp queen
        # 3) player piece touching opp color

        # Rotation
        # 1a) Opp queen
        # 1b) Find opp piece type with least board presence.
        #   If there is only one piece use that
        #   Else Find Center of Mass for the hive
        #   Find piece closest to CoM, if same distance take one and use the other for mirroring

        # 2) Find player piece type with least board presence.
        #    If there is only one piece use that
        #    Else Find Center of Mass for the hive
        #    Find piece closest to CoM, if same distance take one and use the other for mirroring
        # 3) Opp touching piece

        # Mirroring
        # 1a) If a player pieces exist other than queen use those
        # 1b) Elif opp pieces exist use those
        #    If a single piece exist use that.
        #    If multiple pieces exist use the one closest to player queen, and opp queens in case of ties
        #    If still tied, try the one furthest from the queen


        #If the queen is not in play we go through the list of types and find one that only exist one of.
        #If no such exist, we take one that has the lowest count. (this could still be 3)
        #Then we
        #If the queen is not in play, things get complicated,
        # We apply the transl

        if len(hw_idx)>0:
            idx = hw_idx[0][0]
            dx,dy = - hw.qr[idx]
        else:
            dx,dy = 0,0

        hw.qr[mw,0] += dx
        hw.qr[mw,1] += dy
        hb.qr[mb,0] += dx
        hb.qr[mb,1] += dy

        #Now rotation
        if len(hb_idx) > 0:
            idx = hb_idx[0][0]
            hb_qrs = self.axial_to_cube(hb.qr)
            rots = self.all_rotations(hb_qrs[idx])
            neg = rots[:,0] < 0
            n = (neg.nonzero()[-1] + 1) % 6
            if n != 0:
                print("here!")

            hb_qrs_r = self.rotate_tensor(hb_qrs,n.item())
            # hb.qr = hb_qrs_r[:,:-1]

            hw_qrs = self.axial_to_cube(hw.qr)
            hw_qrs_r = self.rotate_tensor(hw_qrs,n.item())
            # hw.qr = hw_qrs_r[:,:-1]
            hw.qr[mw,:] = hw_qrs_r[mw,:2]
            hb.qr[mb,:] = hb_qrs_r[mb,:2]


        # Finally mirroring
        if len(hw_idx) > 1:
            qrs_sel = hw_qrs_r[hw_idx[1]]
        elif len(hb_idx) > 1:
            qrs_sel = hb_qrs_r[hb_idx[1]]
        else:
            qrs_sel = None
        if qrs_sel is not None and qrs_sel[0,0] <= 0:
            hw.qr[mw,0] = torch.sign(hw_qrs_r[mw,0])*torch.abs(hw_qrs_r[mw,2])
            hb.qr[mb,0] = torch.sign(hb_qrs_r[mb,0])*torch.abs(hb_qrs_r[mb,2])

            hw.qr[mw,1] = hw_qrs_r[mw,1]
            hb.qr[mb,1] = hb_qrs_r[mb,1]

        #Finally we translate the pieces away from the edge
        self.shift_pieces_from_edges()
        print(f"White after: {hw.qr},{hw.in_play*1}")
        print(f"Black after: {hb.qr},{hb.in_play*1}")
        return







    def all_rotations(self, qrs):
        a = - torch.roll(qrs,1)
        b = torch.roll(qrs,2)
        c = - torch.roll(qrs,3)
        d = torch.roll(qrs,4)
        e = - torch.roll(qrs,5)
        rots = torch.stack((qrs,a,b,c,d,e))
        return rots

    def rotate_tensor(self,t,n=1):
        return (-1)**n * torch.roll(t,n,dims=1)



    def axial_to_cube(self,qr):
        q = qr[:,0]
        r = qr[:,1]
        s = -q-r
        return torch.stack((q,r,s),dim=1)


    def generate_int_form(self):
        """
        Beetle = 1
        Queen = 2
        Grasshopper = 3
        Spider = 4
        Ant = 5

        current_player is positive numbers
        opp is negative numbers

        level=0 *1
        level=-1 *100
        level=-2 *10000
        level=-3 *1000000
        level=-4 *100000000
        level=-5 *10000000000
        """
        b = torch.zeros(self.board_size,dtype=torch.int64)
        hives = [self.hive_player(), self.hive_opp()]
        m = -1
        for hive in hives:
            m *= -1
            for (id,in_play,level,qr) in hive:
                if in_play:
                    val = id+1
                    b[qr[0],qr[1]] += m*val*100**(level)
        return b

    def perform_action(self,action_idx):
        hive = self.hive_player()

        i = action_idx // (self.board_len*self.board_len)
        j = (action_idx // self.board_len) % self.board_len
        k = action_idx % self.board_len

        a = torch.arange(11*24*24)
        b = a.view(11,24,24)
        assert b[i,j,k] == action_idx
        assert hive.moves[i,j,k] == True
        #hive.moves.view(-1).nonzero().squeeze()
        hive.move_piece(i,j,k)
        self.next_player()
        return


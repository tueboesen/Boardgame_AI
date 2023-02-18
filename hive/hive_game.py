import copy
import itertools

import torch

from Templates.game import Game
from hive.hive import Hive
from hive.hive_gamelogic import piece_symbol, generate_board, bitmap_get_neighbors, generate_conv_board, \
    generate_graph, find_moveable_nodes, DIRECTIONS
from hive.movements import calculate_moves
from hive.hive_ui import TransformMatrix


class HiveGame(Game):
    """
    Contains everything happening on the board in hive.
    """
    def __init__(self):
        self.hive_white = Hive(white=True)
        self.hive_black = Hive(white=False)
        self.hives = [self.hive_white, self.hive_black]
        self.turn = 1
        self.whites_turn = True
        self.board_len = len(self.hive_white) + len(self.hive_black) + 2
        self.winner = None
        self._game_over = False
        self.npieces_per_player = len(self.hive_white)
        self.calculate_valid_moves()
        self.canon_actions = None
        self.transform = TransformMatrix()
        self.hist = ['start']

    def __repr__(self) -> str:
        return f"Turn={self.turn},player={'white' if self.whites_turn else 'black'}"

    @property
    def action_size(self):
        return self.npieces_per_player * self.board_len * self.board_len
    @property
    def board_size(self):
        return (self.board_len, self.board_len)

    @property
    def game_over(self):
        return self._game_over

    @property
    def summary(self):
        return f"Turn: {self.turn},   Players turn: {'White' if self.whites_turn else 'Black'}"

    @property
    def current_player(self):
        p = 0 if self.whites_turn else 1
        return p

    @property
    def hive_player(self):
        return self.hive_white if self.whites_turn else self.hive_black

    @property
    def hive_opp(self):
        return self.hive_black if self.whites_turn else self.hive_white

    def reset(self):
        self.hive_white = Hive(white=True)
        self.hive_black = Hive(white=False)
        self.hives = [self.hive_white, self.hive_black]
        self.turn = 1
        self.whites_turn = True
        self.board_len = len(self.hive_white) + len(self.hive_black) + 2
        self.winner = None
        self.npieces_per_player = len(self.hive_white)
        self.calculate_valid_moves()
        self.canon_actions = None
        self.action_size = self.npieces_per_player * self.board_len * self.board_len
        self.transform = TransformMatrix()


    def rep_viz(self, hive):
        qr_viz = hive.qr.clone()
        qr_viz[hive.in_play] = self.transform.inverse(hive.qr[hive.in_play])
        return qr_viz

    def apply_transform(self,A,hive=None):
        if hive is None:
            for hive in self.hives:
                hive.qr[hive.in_play] = self.transform.forward(hive.qr[hive.in_play],A)
        else:
            hive.qr[hive.in_play] = self.transform.forward(hive.qr[hive.in_play], A)


    def nn_rep(self):
        """
        This should always be from the perspective of the player
        """
        bbs = []
        hives = [self.hive_player,self.hive_opp]
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

    def update_board_state(self):
        hive_player = self.hive_player
        hive_opp = self.hive_opp

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
        self.hive_player.bit_state = bit_state_player
        self.hive_opp.bit_state = bit_state_opp
        # generate_graph(self.board_state)

    def calculate_valid_moves(self):
        hive_player = self.hive_player
        hive_opp = self.hive_opp
        self.update_board_state()
        hive_player.moves[:, :, :] = False

        if hive_player.check_played_all():  # Generate list of spawn locations
            spawn_locations = None
        else:
            if hive_player.played_piece():
                bit_player_nn = bitmap_get_neighbors(hive_player.bit_state)
                bit_opp_nn = bitmap_get_neighbors(hive_opp.bit_state)
                spawn_locations = bit_player_nn & (~ (bit_opp_nn | hive_opp.bit_state)) & ~ hive_player.bit_state
            else:
                if hive_opp.played_piece():
                    spawn_locations = torch.roll(hive_opp.bit_state, 1, dims=0) #No reason to give all 6, when they are all the same up to rotation
                else:
                    spawn_locations = generate_board(self.board_len,[(12,12)],bit=True) #No reason to give the whole board

        # Then we go through each piece and see where it can move to
        hive_player.first_of_type[:] = True
        if hive_player.played_queen():
            g, nodes = generate_graph(self.board_state)
            moveable_node_indices = find_moveable_nodes(g)
            moveable_positions = nodes[moveable_node_indices].tolist()
            for i, (id,in_play,level,qr) in enumerate(hive_player):
                if in_play==False and hive_player.first_of_type[id]:
                    hive_player.moves[i,:,:] = spawn_locations
                    hive_player.first_of_type[id] = False
                elif level == 0 and qr.tolist() in moveable_positions:
                    state = self.board_state if piece_symbol(id) == 'b' else self.bit_state
                    hive_player.moves[i,:,:] = calculate_moves(id,qr,state)
            if not hive_player.can_move():
                self.next_player()
                self.calculate_valid_moves()
        elif (hive_player.in_play == True).sum() < 3:
            for i, (id,in_play,level,qr) in enumerate(hive_player):
                if in_play==False and hive_player.first_of_type[id]:
                    hive_player.moves[i,:,:] = spawn_locations
                    hive_player.first_of_type[id] = False
        else:
            hive_player.moves[0,:,:] = spawn_locations
        return hive_player.moves

    def next_player(self):
        """
        Advances the turn to the next player and checks whether the game is over
        :return:
        """
        self.whites_turn = not self.whites_turn
        if self.whites_turn:
            self.turn += 1

    def check_winners(self):
        for hive in self.hives:
            surrounded = True
            if hive.in_play[0]:
                qr = hive.qr[0]
                for direction in DIRECTIONS:
                    qr_nn = qr+direction
                    if self.check_if_coordinate_filled(qr_nn,self.hive_white) or self.check_if_coordinate_filled(qr_nn,self.hive_black):
                        pass
                    else:
                        surrounded = False
                        break
                if surrounded:
                    hive.lost = True
                    self._game_over = True
        if self.hist.count(self.canonical_string_rep()) >= 3:
            self._game_over = True
        #
        # if self.turn >= 100:
        #     self.game_over = True
        #     self.hive_white.lost = True
        if self.game_over:
            if self.hive_white.lost ^ self.hive_black.lost:
                self.winner = 1 if self.hive_white.lost else 0
            else:
                self.winner = None
            return

    def reward(self):
        if self.winner is None:
            reward = 0
        else:
            if self.winner == self.current_player:
                reward = 1
            else:
                reward = -1
        return reward

    def check_if_coordinate_filled(self,qr,hive):
        m1 = hive.qr[:,0] == qr[0]
        m2 = hive.qr[:,1] == qr[1]
        m = m1 & m2
        return m.any()





    def shift_pieces_from_edges(self,add_to_transform=True):
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
            if add_to_transform:
                A = self.matrix_translation(dq,dr,-dq-dr)
                self.transform.update(A)
        return

    def reorder_pieces(self):
        for hive in self.hives:
            if hive.check_played_all():
                continue
            n = torch.arange(hive.npieces)
            a = torch.stack((n,hive.types,hive.in_play),dim=-1)
            b = a[a[:, 2].sort(descending=True)[1]]
            c = b[b[:, 1].sort()[1]]
            permutation = c[:,0]
            hive.qr = hive.qr[permutation,:]
            hive.in_play = hive.in_play[permutation]
        return

    def generate_canonical_board(self):
        """
        This routine performs modulo 60 degree rotations
        and integer translations to the board such that the first white piece is on (0,0)
        and the first black piece is at (x,y),x>0,y>0 (the first quadrant)
        if multiple rotations ends in the first quadrant,
        we take the counterclockwise rotation that ends there but didn't start there.
        Furthermore it mirrors the board along the x-axis such that the first piece in play not residing on the x-axis has a positive y value.

        This does not completely fixate the board since we do not account for piece permutation, but in practice we should at most have 2 identical boards since we are using the first pieces
        of which there only exist 2 of each,
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

        """

        hp = self.hive_player
        ho = self.hive_opp
        hives = [hp,ho]
        self.update_board_state()

        #we need to select 3 pieces for translation, rotation and mirroring.

        np = hp.in_play.sum()
        no = ho.in_play.sum()
        p_idx = hp.in_play.nonzero().squeeze(dim=0)
        o_idx = ho.in_play.nonzero().squeeze(dim=0)

        piece_sel = []
        #We have 4 different strategies, depending on the queens in play:
        if hp.played_queen() and ho.played_queen():
            piece_sel.append((0,0,False))
            piece_sel.append((1,0,False))
            if np > 1:
                piece_sel.append((0,p_idx[1],True))
            elif no > 1:
                piece_sel.append((1,o_idx[1],True))
        elif hp.played_queen():
            piece_sel.append((0,0,False))
            piece_sel.append((1,o_idx[0],True))
            if np > 1:
                piece_sel.append((0,p_idx[1],True))
            elif no > 1:
                piece_sel.append((1,o_idx[1],True))
        elif ho.played_queen():
            piece_sel.append((1,0,False))
            if np >= 2:
                piece_sel.append((0,p_idx[0],True))
                piece_sel.append((0,p_idx[1],True))
            elif np == 1:
                piece_sel.append((0,p_idx[0],True))
                if no > 1:
                    piece_sel.append((1, o_idx[1],True))
        else:
            # Find the two pieces of opposing color that touch
            if np == 0:
                if no > 0:
                    piece_sel.append((1, o_idx[0],False))
            else:
                ho_bit_nn = bitmap_get_neighbors(ho.bit_state)
                hp_coord = ho_bit_nn & hp.bit_state
                idx_sel = hp_coord.nonzero()
                m = (hp.qr == idx_sel).all(dim=1)
                idx_hp = m.nonzero()[0]
                piece_sel.append((0,idx_hp.item(),False))

                hp_bit_nn = bitmap_get_neighbors(hp.bit_state)
                ho_coord = hp_bit_nn & ho.bit_state
                idx_sel = ho_coord.nonzero()
                m = (ho.qr == idx_sel).all(dim=1)
                idx_ho = m.nonzero()[0]
                piece_sel.append((1,idx_ho.item(),False))

                if np>1:
                    if p_idx[0] != idx_hp:
                        piece_sel.append((0, p_idx[0],True))
                    else:
                        piece_sel.append((0, p_idx[1],True))
                elif no>1:
                    if o_idx[0] != idx_ho:
                        piece_sel.append((1, o_idx[0],True))
                    else:
                        piece_sel.append((1, o_idx[1],True))

        # First check the pieces selected for possible beetle problems (two pieces on top of each other)
        qr_set = set()
        piece_sel2 = []
        for piece in piece_sel:
            hive_idx, idx,_ = piece
            hive = hives[hive_idx]
            qr = hive.qr[idx]
            if qr in qr_set:
                idx_alt = None
                hive_alt_idx = None
                for idx in p_idx:
                    if hp.qr[idx] not in qr_set:
                        idx_alt = idx
                        hive_alt_idx = 0
                        break
                for idx in o_idx:
                    if idx_alt is not None:
                        break
                    if ho.qr[idx] not in qr_set:
                        idx_alt = idx
                        hive_alt_idx = 1
                        break
                qr_set.add(hives[hive_alt_idx][idx_alt])
                piece_sel2.append((hive_alt_idx,idx_alt,True))
            else:
                qr_set.add(qr)
                piece_sel2.append(piece)
        piece_sel = piece_sel2

        # Now check the selected pieces for possible permutations
        permutations = []
        for i, (hive_idx, idx,permute_possible) in enumerate(piece_sel):
            hive = hives[hive_idx]
            if permute_possible:
                type_sel = hive.types[idx]
                m = hive.types == type_sel
                sim_types_idx = m.nonzero().squeeze(dim=1)
                in_play_idx = hive.in_play.nonzero().squeeze(dim=1)
                # in_play_list = in_play_idx.tolist() if len(in_play_idx) > 1 else [in_play_idx.item()]
                mutable_set = set(sim_types_idx.tolist()).intersection(in_play_idx.tolist())
                perms = []
                for permute in list(mutable_set):
                    perms.append((hive_idx,permute))
                permutations.append(perms)
            else:
                permutations.append([(hive_idx,idx)])

        permutations_all = itertools.product(*permutations)
        possible_permutations = []
        for idx_sel in permutations_all:
            if len(set(idx_sel)) < len(piece_sel):
                continue
            else:
                possible_permutations.append(idx_sel)


        #Finally we do the actual translation, rotation and mirroring given the indices, while accounting for possible permutations.
        strings = []
        save_transform = True if (len(possible_permutations) == 1) else False

        for hive in self.hives:
            hive.qr_old = hive.qr.clone()
        for idx_sel in possible_permutations:
            for hive in self.hives:
                hive.qr = hive.qr_old.clone()
            self.translate_rotate_mirror_matrix(idx_sel,[(hp,hp.qr),(ho,ho.qr)],save_transform=save_transform)
            string = self.canonical_string_rep()
            strings.append(string)


        if len(possible_permutations) > 1:
            for hive in self.hives:
                hive.qr = hive.qr_old.clone()
            p = range(len(strings))
            indices = sorted(p, key=lambda k: strings[k])
            self.translate_rotate_mirror_matrix(possible_permutations[indices[0]],[(hp,hp.qr),(ho,ho.qr)],save_transform=True)
            # self.reverse_translate_rotate_mirror([(hp,hp.qr_canon),(ho,ho.qr_canon)])
            # assert (hp.qr == hp.qr_canon).all() and (ho.qr == ho.qr_canon).all()
            # self.translate_rotate_mirror(possible_permutations[indices[0]], [(hp, hp.qr_canon), (ho, ho.qr_canon)])
        self.hist.append(strings[0])
        return


    def matrix_translation(self,dx,dy,dz):
        A = torch.eye(4)
        A[:,-1] = torch.tensor([dx,dy,dz,1])
        return A

    def matrix_rotation_z(self,angle):
        """
        Rotation matrix around the z-axis using homogenous coordinates
        angle should be in radians
        """
        A = torch.eye(4)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        A[0,0] = ca
        A[0,1] = -sa
        A[1,0] = sa
        A[1,1] = ca
        return A

    def matrix_rotation(self,n):
        """
        Rotation matrix around the diagonal axis using homogenous coordinates. This is for hexagonal coordinates
        angle should be in radians
        # (-1) ** n * torch.roll(t, n, dims=1)
        """
        A = torch.eye(3)
        Arot = (-1)**n *torch.roll(A,n.item(),dims=0)
        Aext = torch.eye(4)
        Aext[:3,:3] = Arot
        return Aext


    def matrix_mirror(self):
        """
        This matrix will do a mirroring of homogeneous hex coordinates
        """
        A = torch.zeros((4,4))
        A[0,2] = 1
        A[1,1] = 1
        A[2,0] = 1
        A[3,3] = 1
        return A


    def translate_rotate_mirror_matrix(self,hive_and_indices,hives_qr,save_transform=True):
        """
        This computes the matrix that corresponds to the translations, rotations and mirrorings needed, such that:
            piece 0 is at origo
            piece 1 is in the first position located in the first quadrant, when rotated counter clockwise.
            piece 2 is mirrored such that q>0

        hives_qr = [(hive1,qr1_to_rotate),(hive2,qr2_to_rotate)]

        We use homogeneous coordinates in order to easily be able to represent a chain of translations and rotations as a single matrix

        """

        # self.transform.reset()
        # A = torch.eye(4)
        if len(hive_and_indices) > 0:  # Translation
            hive_idx, piece_idx = hive_and_indices[0]
            dx, dy = - hives_qr[hive_idx][1][piece_idx]
            # for hive,qr in hives_qr:
            #     qr[hive.in_play, 0] += dx
            #     qr[hive.in_play, 1] += dy
            # self.board_canon.append(('T',(dx,dy)))
            A = self.matrix_translation(dx,dy,-dx-dy)
            self.apply_transform(A)
            if save_transform:
                self.transform.update(A)

        if len(hive_and_indices) > 1:  # Rotation
            hive_idx, piece_idx = hive_and_indices[1]
            qr_sel = hives_qr[hive_idx][1][piece_idx]
            qrs_sel = self.axial_to_cube(qr_sel)
            rots = self.all_rotations(qrs_sel)
            neg = rots[:, 0] < 0
            n = ((neg.nonzero()[-1] + 1) % 6)
            # for hive, qr in hives_qr:
            #     qr_ip = qr[hive.in_play]
            #     qrs = self.axial_to_cube(qr_ip)
            #     qrs_rot = self.rotate_tensor(qrs, n)
            #     qr[hive.in_play, :] = qrs_rot[:, :2]
            # self.board_canon.append(('R',n))
            A = self.matrix_rotation(n)
            self.apply_transform(A)
            if save_transform:
                self.transform.update(A)
        if len(hive_and_indices) > 2:  # Mirroring
            hive_idx, piece_idx = hive_and_indices[2]
            qr_sel = hives_qr[hive_idx][1][piece_idx]
            if qr_sel[1] < 0:
                A = self.matrix_mirror()
                self.apply_transform(A)
                if save_transform:
                    self.transform.update(A)
        self.shift_pieces_from_edges(add_to_transform=save_transform)
        return


    def translate_rotate_mirror(self,hive_and_indices,hives_qr):
        """
        This translates rotates and mirrors the pieces such that:
            piece 0 is at origo
            piece 1 is in the first position located in the first quadrant, when rotated counter clockwise.
            piece 2 is mirrored such that q>0

        hives_qr = [(hive1,qr1_to_rotate),(hive2,qr2_to_rotate)]
        """
        #First we copy the piece positions from qr to qr_canon
        # for hive in self.hives:
        #     hive.qr_canon = hive.qr.clone()
        self.board_canon = []

        if len(hive_and_indices) > 0:  # Translation
            hive_idx, piece_idx = hive_and_indices[0]
            dx, dy = - hives_qr[hive_idx][1][piece_idx]
            for hive,qr in hives_qr:
                qr[hive.in_play, 0] += dx
                qr[hive.in_play, 1] += dy
            self.board_canon.append(('T',(dx,dy)))

        if len(hive_and_indices) > 1:  # Rotation
            hive_idx, piece_idx = hive_and_indices[1]
            qr_sel = hives_qr[hive_idx][1][piece_idx]
            qrs_sel = self.axial_to_cube(qr_sel)
            rots = self.all_rotations(qrs_sel)
            neg = rots[:, 0] < 0
            n = ((neg.nonzero()[-1] + 1) % 6).item()
            for hive, qr in hives_qr:
                qr_ip = qr[hive.in_play]
                qrs = self.axial_to_cube(qr_ip)
                qrs_rot = self.rotate_tensor(qrs, n)
                qr[hive.in_play, :] = qrs_rot[:, :2]
            self.board_canon.append(('R',n))

        if len(hive_and_indices) > 2:  # Mirroring
            hive_idx, piece_idx = hive_and_indices[2]
            qr_sel = hives_qr[hive_idx][1][piece_idx]
            if qr_sel[1] < 0:
                for hive, qr in hives_qr:
                    qr_ip = qr[hive.in_play]
                    qrs = self.axial_to_cube(qr_ip)
                    qr_ip[:, 0] = qrs[:,2]
                    # qr_ip[:, 1] = qrs[]
                    # qr_ip[:, 0] = torch.sign(qrs[:, 0]) * torch.abs(qrs[:, 2])
                    # qr_ip[:, 1] = qrs[:, 1]
                self.board_canon.append(('M'))
        self.shift_pieces_from_edges()
        return

    def reverse_translate_rotate_mirror(self,hives_qr):
        """
        This routine performs the reverse operation of the translate_rotate_mirror routine on the hives_qr given as input
        hives_qr = [(hive1,qr1_to_rotate),(hive2,qr2_to_rotate)]
        """
        # First we translate the pieces back to the edges
        if self.board_canon is None:
            return
        actions = copy.deepcopy(self.board_canon)
        while len(actions)>0:
            a = actions.pop()
            if a[0] == 'T': # Translation
                dx, dy = a[1]
                for hive, qr in hives_qr:
                    qr[hive.in_play, 0] -= dx
                    qr[hive.in_play, 1] -= dy
            elif a[0] == 'R': # Rotation
                n = a[1]
                n = n % 6
                for hive, qr in hives_qr:
                    qr_ip = qr[hive.in_play]
                    qrs = self.axial_to_cube(qr_ip)
                    qrs_rot = self.rotate_tensor(qrs, 6 - n)
                    qr[hive.in_play, :] = qrs_rot[:, :2]
            elif a[0] == 'M': # Mirroring
                for hive, qr in hives_qr:
                    qr_ip = qr[hive.in_play]
                    qrs = self.axial_to_cube(qr_ip)
                    qr_ip[:, 0] = qrs[:, 2]
        return

    def canon_board_to_viz_board(self):
        """
        This routine convert the canon boardstate back into the vizualization boardstate
        """
        for hive in self.hives:
            hive.qr_viz = hive.qr.clone()
            hive.qr_viz[hive.in_play] = self.transform.inverse(hive.qr_viz[hive.in_play])



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
        q = qr[...,0]
        r = qr[...,1]
        s = -q-r
        return torch.stack((q,r,s),dim=-1)



    def canonical_string_rep(self):
        hives = [self.hive_player, self.hive_opp]
        string = ""
        for hive in hives:
            string = string + str(hive.qr[hive.in_play].view(-1).tolist()) + str(hive.types[hive.in_play].tolist()) + str(hive.level[hive.in_play].tolist())
        return string

    def get_valid_moves(self):
        moves = self.hive_player.moves
        return moves.view(-1).nonzero()[:,0]

    def move_piece(self, idx, q, r):
        hp = self.hive_player
        if piece_symbol(hp.types[idx]) == 'b' and hp.in_play[idx]:
            # First we find any potential pieces on the tile we are about to move to,
            # and lower them 1 level and setting pieces_under beetle to that amount
            move_dst = torch.tensor((q, r))
            move_src = hp.qr[idx].clone()

            hp.pieces_under[idx] = 0
            for i, hive in enumerate(self.hives):
                qr = hive.qr[hive.in_play]
                m = (qr == move_dst).all(dim=1)
                hp.pieces_under[idx] += m.sum()

                idx_in_play = hive.in_play.nonzero()
                hive.level[idx_in_play[m]] -= 1

            # Then we move the beetle to that coordinate
            hp.move_piece(idx, q, r)

            # Then we check the coordinates the beetle were at originally and raise the level of all pieces there by one.
            # indices_all = self.find_matching_coordinates(move_src.tolist())
            for i, hive in enumerate(self.hives):
                qr = hive.qr[hive.in_play]
                m = (qr == move_src).all(dim=1)
                # hive.level[hive.in_play][m] += 1
                idx_in_play = hive.in_play.nonzero()
                hive.level[idx_in_play[m]] += 1

        else:
            hp.move_piece(idx, q, r)


    def perform_action(self,action_idx):
        """
        Everytime an action is taken the following things need to happen:
        Push pieces away from edge
        Check if game is over
        Generate new possible moves and determine next player
        Generate a canonical representation of the boardstate as string
        Generate a canonical representation of the boardstate for the neural network

        """
        hive = self.hive_player

        i = action_idx // (self.board_len*self.board_len)
        j = (action_idx // self.board_len) % self.board_len
        k = action_idx % self.board_len
        a = torch.arange(11*24*24)
        b = a.view(11,24,24)
        assert b[i,j,k] == action_idx
        assert hive.moves[i,j,k] == True
        self.move_piece(i,j,k)
        self.next_player()
        self.shift_pieces_from_edges()
        self.generate_canonical_board()
        # if self.viz:
        #     self.canon_board_to_viz_board()
        self.calculate_valid_moves()
        self.check_winners()

        return



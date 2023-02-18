import itertools
import typing

import torch
import igraph as ig


PieceType = int
BoolTensor = torch.BoolTensor
PIECE_TYPES = [QUEEN, SPIDER, BEETLE, GRASSHOPPER, ANT] = range(0,5)

PIECE_SYMBOLS = ["q", "s", "b", "g", "a"]
PIECE_NAMES = ["queen", "spider", "beetle", "grasshopper", "ant"]

PIECES_PER_PLAYER = ["q", 's', 's', 'b', 'b', 'g', 'g', 'g', 'a', 'a', 'a']

DIRECTIONS = torch.tensor([(-1, 0), (1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)])


# PIECES_PER_PLAYER = ["q", 'b', 'b', 'a', 'a']
# ICONS = ['./icons/bee.png', './icons/beetle.png', './icons/grasshopper.png', './icons/spider.png', './icons/ant.png']



def piece_symbol(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_SYMBOLS[piece_type])


def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])


def piece_id(piece_type: PieceType) -> int:
    return PIECE_SYMBOLS.index(piece_type)


def generate_conv_board(size: int, pos:tuple[int,int]=None,ones:bool=False):
    """
    Creates a Tensor bitmap board and inserts +1 on each indices position.

    :param size: size of board
    :param indices: If given, that particular (row,col) will have value +1.
    :return:
    """
    dtype = torch.float
    if ones:
        board_state = torch.ones(size, size, dtype=dtype)
    else:
        board_state = torch.zeros(size, size, dtype=dtype)
    if pos is not None:
        board_state[pos[0], pos[1]] = board_state[pos[0], pos[1]] + 1
    return board_state


def generate_board(size: int, indices: list[tuple[int, int]] = None, bit: bool = True):
    """
    Creates a Tensor bitmap board and inserts +1 on each indices position.

    :param size: size of board
    :param indices: If given, that particular (row,col) will have value +1.
    :param bit: if False, then it returns an int board instead, which is usefull for the beetle that needs to know the number of pieces stacked on a tile.
    :return:
    """
    dtype = torch.bool if bit else torch.int8
    board_state = torch.zeros(size, size, dtype=dtype)
    if indices is not None:
        for row, col in indices:
            board_state[row, col] = board_state[row, col] + 1
    return board_state


def bitmap_get_neighbors(bitboard: BoolTensor) -> BoolTensor:
    """
    Given a board it rolls the board in all 6 hexagonal directions and adds them up.
    For this to work as intended it is important that no True value is ever located on the edge of the bitboard.
    :param bitboard:
    :return:
    """
    assert (bitboard.dtype == torch.bool)
    bitboard1 = torch.roll(bitboard, 1, dims=0)
    bitboard2 = torch.roll(bitboard, 1, dims=1)
    bitboard3 = torch.roll(bitboard, -1, dims=0)
    bitboard4 = torch.roll(bitboard, -1, dims=1)
    bitboard5 = torch.roll(bitboard1, -1, dims=1)
    bitboard6 = torch.roll(bitboard3, 1, dims=1)
    bitboard_nn = bitboard1 | bitboard2 | bitboard3 | bitboard4 | bitboard5 | bitboard6
    return bitboard_nn


def remove_chokepoint_move_indices(move_indices: list[tuple[int, int]], row: int, col: int, bitboard_all: BoolTensor) -> (list[tuple[int, int]], list[tuple[int, int]]):
    """
    Removes chokepoint move_indices.
    Also known as the Freedom to move rule in Hive: https://boardgamegeek.com/wiki/page/Hive_FAQ#toc12

    This routine removes move indices from move_indices where the two positions (row,col) and (move_indices[i][0],move_indices[i][1]) share two neighbours that exist.
    These two neighbors are then effectively choking / preventing the piece to slide from (row,col) -> (move_indices[i][0],move_indices[i][1])

    Note that this routine only works with bitboards, and thus will not work with beetle moves, that require a full board.
    For beetles make sure to use remove_chokepoint_moves, which this method also calls.
    :param move_indices: A list of position indices exactly one move away from (row,col)
    :param row: the starting position (row,col)
    :param col: the starting position (row,col)
    :param bitboard_all: A bitboard of all the pieces in play
    :return: (allowed_move_indices, prohibited_moves)
    """
    bs = bitboard_all.shape[0]
    moves = generate_board(bs, move_indices)
    moves, prohibited_indices = remove_chokepoint_moves(moves, row, col, bitboard_all)
    move_indices = moves.nonzero().tolist()
    move_indices = [(i[0], i[1]) for i in move_indices]
    return move_indices, prohibited_indices


def remove_chokepoint_moves(moves: BoolTensor, row: int, col: int, board_all: torch.Tensor, beetle: bool = False) -> BoolTensor:
    """
    Removes chokepoint moves.
    Also known as the Freedom to move rule in Hive: https://boardgamegeek.com/wiki/page/Hive_FAQ#toc12

    This routine removes moves where the two positions (row,col) and (moves[i,j]>0) share two neighbours that exist.
    This routine works with both int_boards and bit_boards, and int_boards should be used for beetles in order to get correct movements.
    :param moves: A BoolTensor with True on every possible move, each move should be exactly one move away from (row,col)
    :param row: the starting position (row,col)
    :param col: the starting position (row,col)
    :param board_all: A bitboard or intboard of all the pieces in play
    :param beetle: There a special rules for a beetle that allows it to move even if two neighbors are present, this deals with that situation.
    :return: allowed moves, removed_indices.
    """
    bs = board_all.shape[0]
    bit_nn_piece = bitmap_get_neighbors(generate_board(bs, [(row, col)]))
    move_indices = moves.nonzero().tolist()
    removed_indices = []
    for row_move, col_move in move_indices:
        bit_nn_move = bitmap_get_neighbors(generate_board(bs, [(row_move, col_move)]))
        bit_choke_pos = bit_nn_piece & bit_nn_move
        bit_colisions = board_all.bool() & bit_choke_pos
        if bit_colisions.sum() == 2:
            if beetle:  # Beetles have additional checks, we could just let a normal piece check for this as well, but it is faster to skip
                colision_indices = bit_colisions.nonzero().tolist()
                col_height = min(board_all[colision_indices[0][0], colision_indices[0][1]], board_all[colision_indices[1][0], colision_indices[1][1]])
                piece_height = max(board_all[row, col] - 1, board_all[row_move, col_move])
                if col_height > piece_height:
                    moves[row_move, col_move] = 0
                    removed_indices.append((row_move, col_move))
            else:
                moves[row_move, col_move] = 0
                removed_indices.append((row_move, col_move))
    return moves, removed_indices


def find_moveable_nodes(g: ig.Graph) -> list[int]:
    """
    Given a graph, it finds the nodes in that graph that could be cut while keeping the remaining graph connected.
    In practice this is done by generating all bridge edges in the graph and removing the bridge edges that only has degree 1 (leaf nodes).
    The moveable nodes are then the nodes that are not bridge nodes, except if they are leaf nodes, since leaf nodes are always moveable.
    :param g:
    :return: moveable node indices
    """
    nodes = g.vcount()
    articulation_points = g.articulation_points()
    moveable_node_indices = set(range(nodes)).difference(articulation_points)
    return list(moveable_node_indices)


def axial_distance_fast(x: torch.Tensor) -> torch.Tensor:
    """
    Finds the distance between two hexes.
    See the distance function for axial grids: https://www.redblobgames.com/grids/hexagons/
    :param hexes_pair:
    :return:
    """
    if x.shape[0] > 0:
        dists = (torch.abs(x[:,0,0]-x[:,1,0]) + torch.abs(x[:,0,0] + x[:,0,1] - x[:,1,0] - x[:,1,1]) + torch.abs(x[:,0,1] - x[:,1,1])) / 2
    else:
        dists = torch.tensor([])
    return dists


def generate_graph(board_state: torch.Tensor) -> (ig.Graph, torch.Tensor):
    """
    Generates a graph given the board_state.

    :param bitboard_state:
    :return: Graph of bitboard_state, and the associated node position indices
    """
    bs = board_state.clone()
    node_indices = bs.nonzero()
    mask = bs != 0
    if mask.any():
        bs[mask] -= 1
        node_indices2 = bs.nonzero()
        if len(node_indices2)>0:
            node_indices = torch.cat((node_indices,node_indices2))

    nodes = len(node_indices)
    # node_indices_axial = self.offset_to_axial(node_indices)
    # node_pairs = list(itertools.combinations(node_indices, 2))
    indices = torch.combinations(torch.arange(nodes),2)
    node_pairs = node_indices[indices]

    # distances = axial_distance(node_pairs)
    distances = axial_distance_fast(node_pairs)
    mask = distances < 1.1
    node_combinations = torch.combinations(torch.arange(nodes), 2)
    edges = node_combinations[mask]
    g = ig.Graph(nodes, edges.tolist())
    if nodes > 0:
        connected_graph = g.is_connected()
        if not connected_graph:
            draw_board(board_state)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ig.plot(g, target=ax)
            assert connected_graph

    return g, node_indices

def axial_to_cube_ext(qr):
    q = qr[...,0]
    r = qr[...,1]
    s = -q-r
    t = torch.ones_like(q)
    return torch.stack((q,r,s,t),dim=-1)

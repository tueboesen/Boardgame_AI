import torch

from hive.HiveGameLogic_utils import piece_name, BoolTensor, generate_board, bitmap_get_neighbors, DIRECTIONS, axial_distance_fast


def calculate_moves(piece_id,qr,state):
    func = eval(f"_moves_{piece_name(piece_id)}")
    moves = func(qr,state)
    return moves
def _moves_queen(qr, bitboard_all: BoolTensor) -> BoolTensor:
    """
    The queen can move only one space per turn. The queen has to obey the freedom to move rule.
    :param bitboard_all:
    :return:
    """
    bs = bitboard_all.shape[0]
    moves = generate_board(bs, [qr])
    moves = bitmap_get_neighbors(moves)
    board_connected = bitmap_of_connected_graph_positions(qr,bitboard_all)
    moves = moves & board_connected  # Only moves next to other pieces
    moves = moves & ~bitboard_all  # Not allowed to move onto other pieces
    moves, _ = remove_chokepoint_moves(moves, qr[0], qr[1], bitboard_all)
    return moves

def _moves_beetle(qr, board_all: torch.tensor) -> torch.tensor:
    """
    The beetle can move only one space per turn.
    The beetle can climb on top of other pieces.
    The beetle has to obey the freedom to move rule.

    Because of its ability to climb on top of other pieces, it needs a board_state, rather than bitboard_state to resolve its possible moves.

    :param board_all: a board detailing the number of pieces at each position on the board
    :return:
    """
    bs = board_all.shape[0]
    moves = generate_board(bs, [qr])
    moves = bitmap_get_neighbors(moves)
    if board_all[qr[0],qr[1]].item() == 1:
        board_connected = bitmap_of_connected_graph_positions(qr,board_all.bool())
        moves = moves & board_connected  # Only moves next to other pieces
    moves, _ = remove_chokepoint_moves(moves, qr[0], qr[1], board_all, beetle=True)
    return moves

def _moves_grasshopper(qr,bitboard_all: BoolTensor) -> BoolTensor:
    """
    Grasshopper: jumps over a line of pieces.
    The grasshopper picks a direction, then jumps in that direction, landing on the first empty space.
    It must jump over at least one piece (of either color, which may also be a grasshopper).
    :param bitboard_all:
    :return:
    """
    directions = DIRECTIONS
    moves = generate_board(bitboard_all.shape[0])
    for i in range(directions.shape[0]):
        direction = directions[i]
        j = 1
        idx = qr + direction * j
        if not bitboard_all[idx[0], idx[1]]:
            continue
        while True:
            j += 1
            idx = qr + direction * j
            if not bitboard_all[idx[0], idx[1]]:
                moves[idx[0], idx[1]] = 1
                break
    return moves

def _moves_spider(qr, bitboard_all: BoolTensor) -> BoolTensor:
    """
    The spider moves three spaces per turn - no more, no less.
    It must move in a direct path and cannot backtrack on itself.
    It may only move around pieces that it is in direct contact with on each step of its move.
    :param bitboard_all:
    :return:
    """
    moves = generate_board(bitboard_all.shape[0])
    bitboard = bitboard_all.clone()
    bitboard[qr[0], qr[1]] = 0
    move_indices_hist = _move_spider_one(bitboard, [tuple(qr.tolist())])
    for move_idx_hist in move_indices_hist:
        move = move_idx_hist[-1]
        moves[move[0], move[1]] = 1
    return moves

def _move_spider_one(bitboard_without_spider: BoolTensor, spider_hist: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    This routine iteratively generates possible spider moves one position away from its current position.
    :param bitboard_without_spider:
    :param spider_hist: A history of the spiders movement, spider_hist[0] was the original starting position,
    while spider_hist[-1] is the most recent position it is attempting to move from.
    :return:
    """
    spider_pos = spider_hist[-1]
    move_indices = []
    nn_origo = []
    for i in range(6):
        direction = DIRECTIONS[i]
        idx = (torch.as_tensor(spider_pos) + direction).tolist()
        if bitboard_without_spider[idx[0], idx[1]]:
            nn_origo.append((idx[0], idx[1]))
        elif (idx[0], idx[1]) not in spider_hist:  # The position is open and not part of the move history
            move_indices.append((idx[0], idx[1]))
    for move in list(move_indices):
        nn_move = []
        for i in range(6):
            direction = DIRECTIONS[i]
            idx = (torch.as_tensor(move) + direction).tolist()
            if bitboard_without_spider[idx[0], idx[1]]:
                nn_move.append((idx[0], idx[1]))
        if not set(nn_move).intersection(nn_origo):  # Are we walking around a piece? Meaning does our end and start position share a neighbour that exist?
            move_indices.remove(move)
    move_indices, _ = remove_chokepoint_move_indices(move_indices, spider_pos[0], spider_pos[1], bitboard_without_spider)

    spider_hist_all = []
    for move in move_indices:
        spider_hist_new = list(spider_hist)
        spider_hist_new.append(move)
        if len(spider_hist_new) < 4:
            spider_hist_all += _move_spider_one(bitboard_without_spider, spider_hist_new)
        else:
            spider_hist_all.append(spider_hist_new)
    return spider_hist_all

def _moves_ant(qr, bitboard_all: BoolTensor,use_fast=True) -> BoolTensor:
    """
    The ant can move from its position to any other position around the hive (restricted by the 'freedom to move' and 'one hive' rules).
    :param bitboard_all:
    :return:
    """
    moves = generate_board(bitboard_all.shape[0])
    if use_fast:
        board = bitboard_all.clone()
        board[qr[0], qr[1]] = 0
        board_nn = bitmap_get_neighbors(board)
        allowed_move_indices = _moves_ant_fast(board, board_nn, qr)
        moves[allowed_move_indices[:,0],allowed_move_indices[:,1]] = 1
    else:
        board = bitboard_all.clone()
        board[qr[0], qr[1]] = 0
        board_nn = bitmap_get_neighbors(board)
        move_idx_set, _ = _move_ant_one(board, board_nn, qr, set(), set())
        if move_idx_set:
            if (qr[0].item(),qr[1].item()) not in move_idx_set:
                move_idx_set, _ = _move_ant_one(board, board_nn, qr, set(), set())
            move_idx_set.remove((qr[0].item(),qr[1].item()))
        move_idx_list = list(move_idx_set)
        for move_idx in move_idx_list:
            moves[move_idx[0], move_idx[1]] = 1
    return moves

def _moves_ant_fast(bitboard_without_ant: BoolTensor, bitboard_without_ant_nn: BoolTensor, position: tuple[int, int]):
    """
    The strategy here is to generate the nn indices of potential move locations and then create a 1 step graph of those.
    Then start from ant origo and traverse to all neighbouring nn locations one at a time recursively, and for each one determine whether it is allowed or not
    """
    bitboard_pos_moves = bitboard_without_ant_nn ^ bitboard_without_ant
    # bitboard_pos_moves[position[0],position[1]] = 1
    p_idx = bitboard_without_ant.nonzero()
    nn_idx = bitboard_pos_moves.nonzero()
    nodes_nn = len(nn_idx)
    nodes_p = len(p_idx)
    indices_nn_p = torch.stack([torch.arange(nodes_nn).repeat(nodes_p),torch.arange(nodes_p).repeat_interleave(nodes_nn)],dim=1)
    indices = torch.combinations(torch.arange(nodes_nn), 2)
    node_pairs_nn_p = torch.stack((nn_idx[indices_nn_p[:,0]],p_idx[indices_nn_p[:,1]]),dim=1)
    node_pairs = nn_idx[indices]
    distances = axial_distance_fast(node_pairs)
    distances_nn_p = axial_distance_fast(node_pairs_nn_p)
    mask = distances < 1.1
    mask_nn_p = distances_nn_p < 1.1
    # node_combinations = torch.combinations(torch.arange(nodes_nn), 2)
    edges = indices[mask]
    edges_nn_p = indices_nn_p[mask_nn_p]
    # node_ant =  position == nn_idx
    node_ant = (position == nn_idx).all(dim=1).nonzero().squeeze().item()
    prohibited_nodes = set([node_ant])
    allowed_nodes = set()
    nodes_checked = set([node_ant])
    new_ant_nodes = []
    while True:
        # node_ant = (position == nn_idx).all(dim=1).nonzero().squeeze()
        edges_to_check = (edges == node_ant).any(dim=1).nonzero()
        nodes_to_check = edges[edges_to_check].view(-1).unique()

        p2 = edges_nn_p[edges_nn_p[:, 0] == node_ant, 1]
        s2 = set(p2.tolist())
        for node in nodes_to_check:
            node_number = node.item()
            if node.item() in nodes_checked:
                continue
            p1 = edges_nn_p[edges_nn_p[:,0] == node,1]
            p_intersect = s2.intersection(p1.tolist())
            if len(p_intersect) >= 2:
                prohibited_nodes.add(node_number)
            else:
                allowed_nodes.add(node_number)
                new_ant_nodes.append(node_number)
            nodes_checked.add(node_number)
        if len(new_ant_nodes) > 0:
            node_ant = new_ant_nodes.pop()
        else:
            break
    allowed_positions = nn_idx[list(allowed_nodes)]
    return allowed_positions


def _move_ant_one(bitboard_without_ant: BoolTensor, bitboard_without_ant_nn: BoolTensor, position: tuple[int, int], move_set: set[tuple[int, int]], prohibited_move_set: set[tuple[int, int]]):
    """
    Recursively builds the possible moves for the ant, prohibited moves, are moves that were removed by the 'freedom to move rule', these prohibited moves
    needs to be kept in order to ensure that the ant doesn't get stuck in a recursive loop trying to get in there.
    We keep the probhibited moves as a separate set so we do not have to check for it every time.
    :param bitboard_without_ant: board_state without the ant
    :param bitboard_without_ant_nn: nearest neighbours of board_state without the ant
    :param position: current ant position
    :param move_set: set of allowed ant move positions
    :param prohibited_move_set: set of prohibited ant move positions
    :return: allowed move positions, prohibited move positions
    """
    move_indices = []
    move_hist_all = move_set.union(prohibited_move_set)
    # nn = DIRECTIONS + position
    # m1 = bitboard_without_ant[nn[:,0],nn[:,1]]
    # m2 = bitboard_without_ant_nn[nn[:,0],nn[:,1]]
    # nn.tolist() in move_hist_all
    pos = torch.tensor(position)
    for i in range(6):
        direction = DIRECTIONS[i]
        idx = pos + direction
        if not bitboard_without_ant[idx[0], idx[1]] \
                and bitboard_without_ant_nn[idx[0], idx[1]] \
                and (idx[0].item(), idx[1].item()) not in move_hist_all:  # The position is open and is neighbor to another piece and not part of the move history
            move_indices.append((idx[0], idx[1]))
    move_indices, removed_indices = remove_chokepoint_move_indices(move_indices, position[0], position[1], bitboard_without_ant)
    prohibited_move_set.update(removed_indices)
    moves_to_check = []
    for move in move_indices:
        if move not in move_set:
            moves_to_check.append(move)
            move_set.add(move)
    for move in moves_to_check:
        move_set, prohibited_move_set = _move_ant_one(bitboard_without_ant, bitboard_without_ant_nn, move, move_set, prohibited_move_set)
    return move_set, prohibited_move_set

def bitmap_of_connected_graph_positions(qr,bitboard_all):
    """
    Given bitboard_all it generates a bitmap of places that this piece could potentially move to that would keep the 'One hive' rule.
    :param bitboard_all:
    :return:
    """
    bitboard_all_except_piece = bitboard_all.clone()
    bitboard_all_except_piece[qr[0], qr[1]] = 0
    bitmap_nn = bitmap_get_neighbors(bitboard_all_except_piece) | bitboard_all_except_piece
    return bitmap_nn

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
import torch

from hive.HiveGameLogic_utils import PIECES_PER_PLAYER, piece_id, BoolTensor


class Hive:
    """
    Creates a single hive containing all the pieces for 1 player.

    q = y
    r = x
    """

    def __init__(self, white):
        self.npieces = len(PIECES_PER_PLAYER)
        self.qr = torch.empty((self.npieces,2),dtype=torch.int64)
        self.types = torch.empty(self.npieces,dtype=torch.int64)
        self.in_play = torch.zeros(self.npieces,dtype=torch.bool)
        self.level = torch.zeros(self.npieces,dtype=torch.int64)
        self.pieces_under = torch.zeros(self.npieces,dtype=torch.int64)
        self.qr[:,0] = torch.arange(self.npieces)
        self.qr[:,1] = -2 + (not white) * (2 * len(PIECES_PER_PLAYER) + 5)
        self.moves = torch.zeros(self.npieces,2*self.npieces+2,2*self.npieces+2,dtype=torch.bool)
        self._i = 0
        self.lost = False
        self.bit_state = None
        for i, piece in enumerate(PIECES_PER_PLAYER):
            self.types[i] = piece_id(piece)

        self.white = white
        return

    def __repr__(self) -> str:
        return f"{'white' if self.white else 'black'!r}"

    def __str__(self) -> str:
        return f"{'white' if self.white else 'black'}"

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.npieces:
            result = (self.types[self._i], self.in_play[self._i], self.level[self._i], self.qr[self._i])
            self._i += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return self.npieces

    def __getitem__(self, item):
        return (self.types[item], self.in_play[item], self.level[item], self.qr[item])

    def played_piece(self):
        return self.in_play.any()
    def check_played_all(self):
        return self.in_play.all()

    def played_queen(self):
        return self.in_play[0]

    def can_move(self):
        return (self.moves == True).any().item()

    def move_piece(self, idx: int, q: int, r: int):
        """
        Moves a single piece identified by idx, to position (x,y)
        :param idx:
        :param q:
        :param r:
        :return:
        """
        self.qr[idx] = torch.tensor((q, r))
        self.in_play[idx] = True


    # def check_if_coordinate_filled(self,qr,hive):
    #     m1 = hive.qr[:,0] == qr[0]
    #     m2 = hive.qr[:,1] == qr[1]
    #     m = m1 & m2
    #     return m.any()


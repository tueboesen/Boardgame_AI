import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int,dt=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)
        self.dt = dt

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """

    def forward(self, x):
        x = x + self.dt * self.convblock(x)  # skip connection
        return x


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeNNet, self).__init__()

        # game params
        nlayers = 3
        self.nlayers = nlayers
        self.board_x, self.board_y = game.board_size
        self.action_size = game.action_size
        self.args = args
        self.in_channels = 1
        self.out_channels = 1

        self.conv_open = nn.Conv2d(self.in_channels, args.num_channels, 3, stride=1, padding=1)
        self.resblocks = nn.ModuleList()
        self.policy_block = nn.ModuleList()

        for i in range(nlayers):
            self.resblocks.append(ResBlock(args.num_channels,args.num_channels,args.num_channels))

        self.p1 = nn.Conv2d(args.num_channels, args.num_channels, 1)
        self.p2 = nn.ReLU()
        self.p3 = nn.Conv2d(args.num_channels, self.out_channels, 1,bias=True)

        self.v1 = nn.Conv2d(args.num_channels, 1, 1)
        self.v2 = nn.ReLU()
        self.v3 = nn.Linear(self.board_x*self.board_y,256)
        self.v4 = nn.ReLU()
        self.v5 = nn.Linear(256,1)






    def forward(self, s):
        s = s.view(-1, self.in_channels, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = self.conv_open(s)
        for block in self.resblocks:
            s = block(s)

        # Policy
        p = self.p1(s)
        p = self.p2(p)
        p = self.p3(p)
        p = p.view(-1,self.action_size)

        # Value
        v = self.v1(s)
        # v = self.v2(v)
        v = v.view(-1,self.action_size)
        v = self.v3(v)
        # v4 = self.v4(v)
        v = self.v5(v)
        #
        #
        #
        # pi = self.fc1(s)
        # # s = torch.cat((s,pi),dim=1)
        # v = self.fc2(s)                                                                         # batch_size x action_size
        # v = self.fc3(v)
        #
        # # s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        # #
        # # pi = self.fc3(s)                                                                         # batch_size x action_size
        # # v = self.fc4(s)                                                                          # batch_size x 1
        # F.log_softmax(p, dim=1)
        return p, torch.tanh(v)

# class TicTacToeNNet(nn.Module):
#     def __init__(self, game, args):
#         super().__init__()
#         self.dl1 = nn.Linear(9, 36)
#         self.dl2 = nn.Linear(36, 36)
#         self.output_layer = nn.Linear(36, 9)
#
#         self.dl3 = nn.Linear(36,1)
#
#     def forward(self, x):
#         x = x.view(-1,9)
#         x = self.dl1(x)
#         x = torch.relu(x)
#
#         x = self.dl2(x)
#         x = torch.relu(x)
#
#         p = self.output_layer(x)
#         p = F.log_softmax(p, dim=1)
#         # x = torch.sigmoid(x)
#         v = self.dl3(x)
#         v = torch.tanh(v)
#         return p, v
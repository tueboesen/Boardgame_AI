import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiveNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.board_size
        self.action_size = game.action_size
        self.args = args
        self.in_channels = 22
        self.out_channels = 8

        super(HiveNNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(args.num_channels, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, self.out_channels, 3, stride=1, padding=1)


        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(self.out_channels)

        self.fc1 = nn.Linear(self.out_channels*(self.board_x)*(self.board_y), self.action_size)
        self.fc2 = nn.Linear(self.out_channels*(self.board_x)*(self.board_y), 1)
        # self.fc_bn1 = nn.BatchNorm1d(6*self.action_size)
        #
        # self.fc_bn2 = nn.BatchNorm1d(2*self.action_size)
        #
        # self.fc3 = nn.Linear(2*self.action_size, self.action_size)
        #
        # self.fc4 = nn.Linear(self.action_size, 1)
        #pytorch_total_params = sum(p.numel() for p in model.parameters())



    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.in_channels, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn5(self.conv5(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn6(self.conv6(s)))                          # batch_size x num_channels x board_x x board_y
        # pi = self.conv5(s)

        s = s.view(-1, self.out_channels*(self.board_x)*(self.board_y))
        pi = self.fc1(s)
        # s = torch.cat((s,pi),dim=1)
        v = self.fc2(s)                                                                         # batch_size x action_size

        # s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        #
        # pi = self.fc3(s)                                                                         # batch_size x action_size
        # v = self.fc4(s)                                                                          # batch_size x 1

        return pi, torch.tanh(v)

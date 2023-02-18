import os
import sys
import time

import numpy as np
from torch import nn
from tqdm import tqdm

sys.path.append('../../')
from src.utils import *

import torch
import torch.optim as optim
import torch.nn.functional as F



class NNetWrapper:
    def __init__(self, game,neuralnet,args):
        self.nnet = neuralnet
        pytorch_total_params = sum(p.numel() for p in self.nnet.parameters())
        print(f"Number of parameters: {pytorch_total_params}")
        self.board_x, self.board_y = game.board_size
        self.action_size = game.action_size
        self.args = args
        self.loss_policy = nn.CrossEntropyLoss()
        # self.loss_policy2 = nn.KLDivLoss()
        self.loss_value = nn.MSELoss(reduction='none')
        self.loss_value2 = nn.MSELoss()
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(),lr=self.args['lr'])
        # examples = examples[2:4]

        for epoch in range(self.args.epochs):
            # print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            total_losses = AverageMeter()

            batch_count = max(int(len(examples) / self.args.batch_size),1)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                # sample_ids = np.arange(len(examples))
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards_nn = torch.stack(boards,dim=0)
                target_pis = torch.stack(pis, dim=0)
                target_vs = torch.FloatTensor(vs)


                # predict
                if self.args.cuda:
                    boards_nn, target_pis, target_vs = boards_nn.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards_nn)

                # l_pi = self.loss_policy(out_pi,target_pis)

                # out_pi_norm = F.log_softmax(out_pi,dim=1)
                l_pi = self.loss_policy(out_pi,target_pis)
                prob = torch.softmax(out_pi,dim=1)
                # l_v = self.loss_value(out_v.squeeze(),target_vs)
                l_v = self.loss_value2(out_v.squeeze(),target_vs)

                # l_v_mean = l_v.mean()
                # l_pi = self.loss_pi(target_pis, out_pi)
                # l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards_nn.size(0))
                v_losses.update(l_v.item(), boards_nn.size(0))
                total_losses.update(total_loss, boards_nn.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss=total_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        return

    def predict(self, board_state_nn):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        if self.args['cuda']: board_state_nn = board_state_nn.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_state_nn)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        pi = torch.softmax(pi,dim=1)
        return pi[0], v[0].item()
        # return torch.exp(pi)[0], v[0].item()
        # return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    # def loss_pi(self, targets, outputs):
    #     return -torch.sum(targets * outputs) / targets.size()[0]

    # def loss_v(self, targets, outputs):
    #     return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

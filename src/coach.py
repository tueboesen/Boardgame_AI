import copy
import logging
import os
import pickle
import sys
import glob
from collections import deque
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm

from src.arena import Arena
from src.mcts import MCTS
from src.players import MCTSNNPlayer
from src.utils import get_file

log = logging.getLogger(__name__)

from matplotlib import pyplot as plt

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet.
    """

    def __init__(self, game, nnet, display=None,args=None):
        self.game = game
        self.nnet = nnet
        self.pnet = copy.deepcopy(nnet)
        self.args = args
        self.coach_args = args.coach
        self.mcts_args = args.mcts
        self.display = display
        self.mcts = MCTS(self.nnet,self.mcts_args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.self_play_winners = []
        self.self_play_draws = 0
        self.wins_champion = []
        self.start_iter = 0
        self.draws = []
        self.use_dummy_nn_for_first_iteration = True
        if self.args.load_previous:
            self.mcts.load(self.args.folder)


    def self_play_game(self):
        """
        This function self-plays one game.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        game = self.game
        episodeStep = 0
        # self.mcts.reset()

        while True:
            episodeStep += 1

            move_prob, action, game_next = self.mcts.compute_action(game)

            move_prob_ext = torch.zeros(game.action_size,dtype=torch.float)
            move_prob_ext[game.get_valid_moves()] = move_prob
            trainExamples.append([game.nn_rep(), move_prob_ext, game.current_player])
            game = copy.deepcopy(game_next)


            if game.game_over:
                r = game.reward
                trainExamples = [(x[0], x[1], r[x[2]] * 0.9 ** i) for i,x in enumerate(reversed(trainExamples))]
                if game.winner is None:
                    self.self_play_draws += 1
                else:
                    self.self_play_winners[game.winner] += 1
                return trainExamples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.start_iter, self.coach_args.numIters):
            log.info(f'Starting Iter #{i} ...')
            if self.use_dummy_nn_for_first_iteration and i == 0:
                self.mcts.use_dummy_nn = True
            else:
                self.mcts.use_dummy_nn = False
            self.self_play_winners = [0, 0]
            self.self_play_draws = 0
            iterationTrainExamples = deque([], maxlen=self.coach_args.maxlenOfQueue)
            if not self.skipFirstSelfPlay:
                for _ in tqdm(range(self.coach_args.numEps), desc="Self Play"):
                    iterationTrainExamples += self.self_play_game()
                self.trainExamplesHistory.append(iterationTrainExamples)
                print(f"Self-play winners: player 0: {self.self_play_winners[0]}, player 1: {self.self_play_winners[1]}, draws: {self.self_play_draws}")
            if len(self.trainExamplesHistory) > self.coach_args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            # shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.folder, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.folder, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, mcts_param=self.mcts_args, description='previous', play_to_win=True)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, mcts_param=self.mcts_args, description='new', play_to_win=True)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            player_champion = MCTSNNPlayer(pmcts,description='champion')
            player_contender = MCTSNNPlayer(nmcts,description='contender')
            players = [player_contender, player_champion]
            arena = Arena(players, self.game, self.display)
            wins, draws = arena.playGames(self.coach_args.arenaCompare)


            log.info(f'Wins (contender/champion): {wins[0]}/{wins[1]} - draws {draws}')
            if (sum(wins) == 0) or ((wins[0] / sum(wins)) < self.coach_args.updateThreshold):
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.folder, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.folder, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.folder, filename='best.pth.tar')
                log.info('Flushing training examples')
                self.trainExamplesHistory = []
                self.mcts = MCTS(self.nnet, mcts_param=self.mcts_args)  # reset search tree

            self.mcts.save(self.args.folder,self.getCheckpointFile(i))
            self.wins_champion.append(wins[1])
            self.draws.append(draws)
            self.plot_win_loss()
            self.skipFirstSelfPlay = False

    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration:03d}'

    def saveTrainExamples(self, iteration):
        folder = self.args.folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        torch.save(self.trainExamplesHistory,filename)
        with open(filename, "wb+") as f:
            pickle.dump([iteration, self.trainExamplesHistory], f)
        return


    def loadTrainExamples(self,folder,filename='', ext='.examples'):
        file = get_file(folder, filename, ext)
        if not os.path.isfile(file):
            log.warning(f'File "{file}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info(f"Training examples found, loading: {file}")
            # self.trainExamplesHistory = torch.load(file)
            with open(file, "rb") as f:
                iteration, self.trainExamplesHistory = pickle.load(f)
            self.start_iter = iteration
            self.skipFirstSelfPlay = True

    def plot_win_loss(self):
        plt.figure(1)
        plt.clf()
        # f, (ax1) = plt.subplots(1, 1)
        ngames = self.coach_args.arenaCompare
        losses = [ngames-wins-draws for wins,draws in zip(self.wins_champion,self.draws)]
        plt.plot(self.wins_champion)
        plt.plot(losses)
        plt.plot(self.draws)
        plt.legend(["wins",'loss','draws'])
        plt.pause(0.5)



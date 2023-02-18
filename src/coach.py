import copy
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm

from src.arena import Arena
from src.mcts import MCTS
from src.players import MCTSNNPlayer

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet.
    """

    def __init__(self, game, nnet, args=None,display=None):
        self.game = game
        self.nnet = nnet
        self.pnet = copy.deepcopy(nnet)
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.display = display
        self.mcts = MCTS(self.nnet)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.self_play_winners = []
        self.self_play_draws = 0

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
        self.mcts.reset()

        while True:
            episodeStep += 1

            # canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # pi = self.mcts.getActionProb(board, temp=temp)
            move_prob, action, game_next = self.mcts.compute_action(game)

            move_prob_ext = torch.zeros(game.action_size,dtype=torch.float)
            move_prob_ext[game.get_valid_moves()] = move_prob
            trainExamples.append([game.nn_rep(), move_prob_ext, game.current_player])

            # action_idx = np.random.choice(len(pi), p=pi)
            # board = self.game.getNextState_from_possible_actions(board, action_idx)
            game = copy.deepcopy(game_next)


            if game.game_over:# != 0:
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

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            self.mcts = MCTS(self.nnet)  # reset search tree
            self.self_play_winners = [0, 0]
            self.self_play_draws = 0
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                c = 0
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    iterationTrainExamples += self.self_play_game()
                    # c += 1
                    # if c>1:
                    #     quit()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
            print(f"Self-play winners: player 0: {self.self_play_winners[0]}, player 1: {self.self_play_winners[1]}, draws: {self.self_play_draws}")
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet,rep='previous',play_to_win=True)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet,rep='new',play_to_win=True)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            player_champion = MCTSNNPlayer(self.display,pmcts,description='champion')
            player_contender = MCTSNNPlayer(self.display,nmcts,description='contender')
            players = [player_contender, player_champion]
            arena = Arena(players, self.game, self.display)
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            wins, draws = arena.playGames(self.args.arenaCompare)


            log.info(f'Wins (contender/champion): {wins[0]}/{wins[1]} - draws {draws}')
            if (sum(wins) == 0) or ((wins[0] / sum(wins)) < self.args.updateThreshold):
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                log.info('Flushing training examples')
                self.trainExamplesHistory = []


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

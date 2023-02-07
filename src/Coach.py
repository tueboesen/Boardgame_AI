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

from src.Arena import Arena
from src.MCTS import MCTS
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
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.display = display
        self.mcts = MCTS(self.game, self.nnet)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

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
        # game = self.game
        episodeStep = 0
        game = copy.deepcopy(self.mcts.game)

        while True:
            episodeStep += 1

            # canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # pi = self.mcts.getActionProb(board, temp=temp)
            move_prob, action, game_next = self.mcts.compute_action()

            # move_prob = torch.zeros(self.game.getActionSize(),dtype=torch.float)
            # move_prob[board.get_valid_moves()] = torch.tensor(pi,dtype=torch.float)
            trainExamples.append([game, move_prob, None])

            # action_idx = np.random.choice(len(pi), p=pi)
            # board = self.game.getNextState_from_possible_actions(board, action_idx)
            game = copy.deepcopy(game_next)


            if game.game_over:# != 0:
                r = game.reward()
                # nmoves = 0
                # nvisits = 0
                # for key,val in self.mcts.Vs.items():
                #     nmoves += len(val)
                # for _,val in self.mcts.Ns.items():
                #     nvisits += val+1
                # average_moves = nmoves/len(self.mcts.Vs)
                # print(f"Turn={board.turn}, Winner={board.winner}, average_moves={average_moves:2.2f}, visited_board_states={len(self.mcts.Vs)}, average_board_state_visits={nvisits/len(self.mcts.Ns):2.2f}")
                # # player = 1 if board.whites_turn else -1
                # # raise NotImplementedError("We need to set this up")
                # trainExamples = [(x[0], x[1], r * ((-1) ** (game.whites_turn != x[0].whites_turn))) for x in trainExamples]
                trainExamples = [(x[0], x[1], r * ((-1) ** (game.current_player != x[0].current_player))) for x in trainExamples]

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
            self.mcts = MCTS(self.game, self.nnet)  # reset search tree
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
            pmcts = MCTS(self.game, self.pnet)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            player_champion = MCTSNNPlayer(self.display,pmcts)
            player_contender = MCTSNNPlayer(self.display,nmcts)
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

import logging
import random

import coloredlogs
import numpy as np
import torch

from hive.hive_conf import hive_args
from hive.hive_game import HiveGame
from hive.hive_ui import HiveUI
from hive.nn.hive_nn import HiveNNet
from src.coach import Coach
from src.nnet import NNetWrapper
from src.utils import *
import sys


sys.setrecursionlimit(1000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.



def main():
    log.info('Loading %s...', HiveGame.__name__)
    g = HiveGame()
    display = HiveUI(g)
    log.info('Loading %s...', NNetWrapper.__name__)
    conf = hive_args
    nnet = HiveNNet(g,conf.nn)
    model = NNetWrapper(g,nnet,conf.nn)

    if conf.load_previous:
        log.info(f'Atempting to load checkpoint from folder {conf.folder}')
        file = model.load_checkpoint(conf.folder,conf.model_file)
        log.info(f"Successfully loaded model: {file}")
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, model, display=display, args=conf)

    if conf.load_previous:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples(conf.folder,conf.training_examples_file)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    fix_seed(1334)
    main()

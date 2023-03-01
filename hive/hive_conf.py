import torch

from src.utils import AttrDict

hive_coach_args = AttrDict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 50,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'numItersForTrainExamplesHistory': 20,
})

hive_mcts_args = AttrDict({
    "puct_coefficient": 1.41,
    "num_simulations": 20,
    "temperature": 1.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "add_dirichlet_noise": True,
    "decay": 0.9
})


hive_nn_args = AttrDict({
    'lr': 0.001,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 64,
    'cuda': torch.cuda.is_available()
})


hive_args = AttrDict({
    'load_previous': False,
    'folder': './temp/hive/',
    'model_file': '',
    'training_examples_file': '',

    "coach": hive_coach_args,
    "mcts": hive_mcts_args,
    "nn": hive_nn_args,
})
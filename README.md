
# Framework for boardgames
A general framework made for training neural networks on boardgames. 
The intention with this framework is that it should be easily adaptable to other games, even games with more than two players.

The framework contains the standard policy-maker Monte Carlo Tree Search (MCTS), but also a more advanced Monte Carlo Graph Search (MCGS). 

The framework is still in an early stage and under active development.

## Hive
The was originally created for the boardgame [hive](https://boardgamegeek.com/boardgame/2655/hive), which is a game considered much more difficult than go or chess to solve with a neural network as detailed [here](https://liacs.leidenuniv.nl/~plaata1/papers/IEEE_Conference_Hive_D__Kampert.pdf).
In short the game is made difficult by: translation, rotation and permutation equivariance, combined with computational expensive calculations regarding permitted moves. For more information about my specific hive implementation see the [HiveReadme.md](https://github.com/tueboesen/Hive_nn/blob/master/hive/HiveReadme.md)

![RandomMoves](https://github.com/tueboesen/Hive_nn/blob/master/hive/icons/hive_example.gif)

# Quick start
The fastest way to try out the code is to fork the repository and install the requirements in a poetry environment

Once the requirements have been installed I suggest starting with: play_hive.py, which will start a simple Human versus Random game, and can easily be modified for other play modes.
Note that for games using a neural network, a neural network needs to be trained first using train_hive.py

# Structure
Each game should have its own individual folder.

The src folder contains the framework:
- Players.py contains different types of players including human, random, Monte carlo tree search neural networks.
- Arena.py contains the arena class which takes a number of players and a game and allows them to play games against each other.
- Coach.py contains the self play framework for training a neural network player
- MCTS.py contains the monte carlo tree search algorithm
- MCGS.py contains the monte carlo graph search algorithm

## Monte Carlo Tree Search
Monte carlo tree search (MCTS) is a popular decision-making algorithm for game design, and is fairly easy to implement. The overall idea is that each board state has a number of children corresponding to the possible moves that can be made from the current board state.
MCTS only works well if the number of MCTS simulations is much greater than the average number of possible moves. 
One of the big shortcomings of MCTS is that it does not consider the actual board state and whether it has previously encountered this state. We seek to rectify this with the Monte Carlo Graph Search

For hive games this can be very costly and especially since there is no reusing of old board states. 

## Monte Carlo Graph Search
Monte Carlo Graph Search (MCGS) is a more complicated decision-making algorithm which requires a canonical representation of the board state in order to work. 
MCGS is not very commonly used yet, but shows many promising charactericstics and I believe this will be way to go for a game like Hive.

Typically, MCGS is made using acyclical graphs in order to prevent getting stuck in an infinite circular loop. This is commonly done by including the turn number in the board state. However, this greatly reduces the useability of MCGS since only similar states reached in the same number of turns will be treated as equal.
In my implementation, I take a different approach. I deem a game a draw if the same boardstate is reached 3 times in a single game.
Hence, my canonical string representation includes the number of previous visits of the board state in this particular game.
If a board state has previously been visited it is treated as a pseudo leaf node, meaning that the node is treated as a leaf, but we do not use the neural network to evaluate the boardstate again, instead we take the previous evaluation and devide it by a number (this could be 1+previous number of visits). 
The idea behind this is that if we reach the same board state multiple times, then we are likely nearing a draw situation, and in that case the predicted outcome should be close to 0. 
(1 is a win for the current player, -1 is a win for another player, and 0 is a draw)
This method will enable a consistent MCGS algorithm.

## Play modes
Play modes currently supported are: Human, RandomMoves, MCTS neural network, MCGS neural network. Visualization is handled by pygame.


# To do
The project is still in an early version, among the things left to do are:

- [ ] Update the template Game.py to have all the needed functions
- [ ] Pretrain and save a neural network
- [ ] Gather all hyperparameters in config file, for easy modification 
- [ ] pyx optimization
- [ ] parallelization
- [ ] Add pseudo-rewards to help guide the model, (+- points for locking queens in place, +- values for number of pieces around queens, +- values for locking up pieces?) 


# Credit 
The overall framework is inspired by 
https://github.com/suragnair/alpha-zero-general, while the Monte Carlo Tree Search algorithm is modified from https://github.com/ray-project/ray/blob/master/rllib/algorithms/alpha_zero/mcts.py





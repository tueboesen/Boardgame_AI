# Hive neural network

This is a frame work for training neural networks on board games.

The frame work is still in an early stage and currently only Hive is supported.

- [ ] Fast move calculations
- [ ] Canonical representation of board state
- [ ] Transformation between vizualization board and canonical representation
- [ ] Pygame vizualization and movement selection
- [ ] Human vs human play mode
- [ ] Random player mode
- [ ] Monte carlo tree search algorithm
- [ ] Monte carlo graph search algorithm
- [ ] pyx optimization
- [ ] parallelization

### Board state representation
Hexagonal boards can be represented in many ways as detailed on: https://www.redblobgames.com/grids/hexagons/
For hive the axial configuration is especially suited for storage, and can easily be transformed to cube coordinates or homogenous cube coordinates when needed (https://en.wikipedia.org/wiki/Homogeneous_coordinates).


### Fast move calculations
One of the challenges with training up a neural network to play hive is that the possible moves each piece can move are notoriously complicated to compute.
Especially the ants movement is problematic, since the ant can as far as it wants, which generally enables it to move to any position, except for those locations that it cannot squeeze into.
My first approach ended up taking more than 50% of the total compute time in a MCTS algorithm, but after using graph theory and some optimization I was able to bring this down to around 8%.

### Canonical representation of board state
The canonical representation of the board state is a way to uniquely identify the board state. Meaning that if and only if two board states have the same canonical representation then they are the same board state.
The canonical representation is essential for two reasons:
1) In order to use Monte carlo graph search, a canonical represenation is needed such that we can quickly identify whether a board state has previously been explored or whether it is a new unexplored board state.
2) By using a canonical representation of the board state, we can use a much simpler neural network. Without the canonical representation the neural network would need to be rotational 60 degree equivariant, translational equivariant and permutational equivariant. 

The canonical representation can be made in two different ways.
1) The general way is to use graph automorphism in order to generate a canonical label for the board. The board state can be represented by a graph with colored vertices, where each color represent a piece type, and edge colors, where each edge color represent a distance (note that the distances are discrete since each piece lives on a hexagonal grid). In python such an algorithm can be made using pynauty, which does not support edge colors, but as mentioned in the pynauty documentation this limitation can be circumvented by building several layers of the vertices.  
2) A specific transformation that works for the hive game in particular. This is a complicated algorithm that is too complicated to describe completely here, but in short we select up to 3 pieces in play and use those 3 pieces to uniquely translate, rotate and mirror the board. If the three pieces used for this are not unique, then we test all possible permutation and select the first one after a reordering. 

Both of these methods have their strength and weaknesses. The first method is general, and will work for many other games and it relies on low level code which makes it fast. On the other hand it does not take advantage of any of the clever tricks which we can use to simplify the problem which are specific for this particular game, and it does not actually give a canonical board state, but merely a representation of it. Meaning we still need to make the actual boardstate ourselves if we need it (which we do for the neural network representation)
The second method is specific for this game only, but can use clever tricks to bring down the possible permutations to 2 or 4 if done cleverly. Furthermore, the transformation from current state to canonical representation, can be represented by a (4x4) homogenous coordinate transformation matrix, which is essential for a consistent vizualization of the board.   

### Consistent vizualization of board 
In the previous I mentioned how the board state needs to be represented in a canonical way. However, while the canonical representation is ideal for a neural network representation it is not ideal for vizualization for humans. It can be very hard to play the game when the board is translated, rotated and mirrored in what seems like almost arbitrary fashion after each move. So in order to remedy this we need to have access to a consistent board state that does not rotate or mirror itself after each move. I find that the best way to achieve this is to keep a transformation matrix which contains the total transformations applied to the original boardstate.  
So the transformation matrix $A = A_1 @ A_2 @ ... @ A_n$, where $A_i$ can be either a transformation matrix:

A_trans = [[1 0 0 dx],[0 1 0 dy],[0 0 1 dz],[0 0 0 1]] 

A rotational matrix in homogenous cube coordinates (Note that such a rotational matrix is very different from the standard rotational matrices since this is not a rotation around any of the 3 basis vectors)

A_rot = ?

Or a mirroring matrix in homogenous cube coordinates

A_mirror = 

This transformation matrix allows us to quickly transform the canonical board state and possible moves into a consistent vizualization state, and furthermore the inverse matrix allows us to quickly convert a selected action on the vizualized board back into a selected action on the canonical board. 

One final detail here is that we need to apply additional translations in order to ensure that all the pieces remains within the board, but translations are more acceptable transformations and does not distort the human vizualization in nearly the same degree.

### Vizualization 
My vizualization of hive is made with pygame, which seems suitable for the task. It allows me to easily detect the mouse position and mouse clicks, which can be used to highlight selected and hovered pieces, and highlight possible moves for a selected piece. The vizualization allows an easy overview of the board, and allows a human player to move pieces in an easy and intuitive way.

### Monte carlo tree search
Monte carlo tree search (MCTS) is a popular decision making algorithm for game design, and is fairly easy to implement. The overall idea is that each board state has a number of children corresponding to the possible moves that can be made from the current board state.
MCTS only works reasonably if the number of MCTS simulations are greater than the average number of possible moves. 
For hive games this can be very costly and especially since there is no reusing of old board states. 

### Monte carlo graph search
Monte carlo graph search (MCGS) is a more complicated decision making algorithm which requires a canonical representation of the board state in order to work. MCGS is not very commonly used yet, but shows many promising charactericstics and I believe this will be way to go for a game like Hive.
Typically it is made using acyclical graphs in order to prevent getting stuck in an infinite circular loop. This is commonly done by including the turn number in the board state. However, this greatly reduces the useability of MCGS since only similar states reached in the same number of turns will be treated as equal.
In my implementation I take a different approach. I deem a game a draw if the same boardstate is reached 3 times in a single game.
Hence my canonical string representation includes the number of previous visits of the board state in this particular game.
If a board state has previously been visited it is treated as a pseudo leaf node, meaning that the node is treated as a leaf, but we do not use the neural network to evaluate the boardstate again, instead we take the previous evaluation and devide it by a number (this could be 1+previous number of visits). The idea behind this is that if we reach the same board state multiple times, then we are likely nearing a draw situation, and in that case the predicted outcome should be close to 0. (1 is a win for player 1, -1 is a win for player 2, and 0 is a draw)
This method will enable a consistent MCGS algorithm, with only one remaining problem, namely the search part, where a leaf node is being found. In this state it is important that a temporary search history is kept that also keeps track of any state visited multiple times.

## Installation
Pynauty requires linux, but otherwise the code should be able to run on any system.
## Running it

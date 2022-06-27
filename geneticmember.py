from random import random
import torch
from game import Game
import numpy as np

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

class Member():
    def __init__(self):
        self.dirs = ["up", "left", "down", "right"]
        self.nn = NeuralNetwork().to(device)

    def run(self, board):
        return self.dirs[torch.argmax(self.nn.run(board)).item()]

    # calculate fitness by running the agent on x games and taking the mean score
    def getFitness(self, numAttempts=10):
        meanScore = 0

        for i in range (0, numAttempts):
            game = Game()
            prevBoard = game.board

            while not game.done:
                # run nn on board and pick the desired direction, then make the move
                game.update(self.run(game.board))

                # if the move did not result in the board state to change, make random moves until something happens
                while np.array_equal(game.board, prevBoard):
                    game.update(np.random.choice(self.dirs))

                prevBoard = np.copy(game.board)
        
            meanScore += game.score

        return meanScore / numAttempts

    def breed(self, other, numChildren = 2, mutationChance = 0.1):
        children = []

        for i in range(0, numChildren):
            children.append(Member())

            for index in range(0, 2): # for each weight layer (hidden and output)
                childWeights = np.empty([self.nn.numHidden,16] if index == 0 else [4,self.nn.numHidden])
                selfWeights = self.nn.layers[index].weight
                otherWeights = other.nn.layers[index].weight
                
                for row in range(0, childWeights.shape[0]):
                    for weight in range(0, childWeights.shape[1]):
                        # pick a weight from a random parent
                        childWeights[row, weight] = selfWeights[row][weight].item() if random() < 0.5 else otherWeights[row][weight].item()

                        # mutate randomly, either flip sign or scale
                        if random() < mutationChance:
                            if (random() < 0.5):
                                childWeights[row, weight] *= -1 # flip sign
                            else:
                                childWeights[row, weight] *= random() * 3 # scale by a number from [0,3)

                children[-1].nn.setWeights(index, childWeights)

        return children

# 16-x-4 neural network
class NeuralNetwork(torch.nn.Module):
    def __init__(self, numHidden=8):
        super(NeuralNetwork, self).__init__()
        self.numHidden = numHidden
        self.flatten = torch.nn.Flatten(0, 1) # 4x4 matrix to 1x16 tensor
        self.layers = torch.nn.Sequential(
            self.randomizeWeights(torch.nn.Linear(16, numHidden), [-10,10]),
            self.randomizeWeights(torch.nn.Linear(numHidden, 4), [-10,10]),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=0)
        )

    def randomizeWeights(self, x, bounds): # bounds = [low, high)
        x.weight = torch.nn.Parameter(data=((bounds[1] - bounds[0]) * torch.rand_like(x.weight, device=device) + bounds[0]))
        return x

    # index must be 0 or 1 for hidden layer or output layer respectively
    # weights must be a numpy array of the correct dimensions containing the desired weights
    def setWeights(self, index, weights):
        self.layers[index].weight = torch.nn.Parameter(data=torch.as_tensor(weights, device=device))

    def run(self, x):
        x = torch.as_tensor(x, dtype=torch.float, device=device)
        x = self.flatten(x)
        return self.layers(x)
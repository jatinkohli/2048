import torch
from torch import nn
import numpy as np
from game import Game

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(0, 1) # 4x4 matrix to 1x16 tensor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def run(self, x):
        x = torch.as_tensor(x, dtype=torch.float, device=device)
        x = self.flatten(x)
        return self.linear_relu_stack(x)

def randomizeWeights(x):
    if type(x) == nn.Linear:
        x.weight = nn.Parameter(data=(4 * torch.rand_like(x.weight) - 2)) # [-2,2)
        print(x.weight)

game = Game()
model = NeuralNetwork().to(device)
result = model.run(game.board)
model.apply(randomizeWeights)
print(result)
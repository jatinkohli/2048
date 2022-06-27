import sys
import numpy as np
from baseagent import BaseAgent
from geneticmember import Member

# Agent that plays 2048 by finding an optimal neural network using a genetic algorithm
class GeneticAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prevBoard = []

    def train(self):
        print("TODO")
        
        self.best = Member()

    # determines the best direction (the String "up", "down", "left", or "right")
    def calculate(self, game):
        dir = self.best.run(game.board) # use best agent's neural network

        # if the board did not change states as a result of the last move, pick a random move
        if np.array_equal(game.board, self.prevBoard):
            dir = np.random.choice(self.dirs)

        self.prevBoard = np.copy(game.board)

        return dir

if __name__ == "__main__":
    ga = GeneticAgent()

    ga.train()
    GeneticAgent.main(ga, sys.argv)
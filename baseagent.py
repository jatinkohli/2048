import sys
import numpy as np
from game import Game

class BaseAgent:
    def __init__(self):
        self.dirs = ["up", "left", "down", "right"]

    # Agent that plays 2048 greedily by trying to maximize the score each move

    # determines the best direction (the String "up", "down", "left", or "right") to maximize the score for the given board state
    # in event of a tie, one of the best directions is randomly picked
    def calculate(self, game):
        return np.random.choice(self.dirs)

    def test(self, numAttempts, numTrials, printInfo):
        winBoard = 2048*np.ones((4,4), dtype=np.int16)

        for trial in range(0, numTrials):
            bestScore = 0
            worstScore = 4e6 # higher than the max theoretical score
            meanScore = 0
            numWins = 0 # win is defined as getting a 2048 or higher tile

            if printInfo:
                print(f"Trial {trial}:")

            for i in range (0, numAttempts):
                # if print:
                #     if numAttempts > 10 and i % (int(numAttempts / 10)) == 0:
                #         print(f"{100 * i / numAttempts}% done with testing")
                #     print(i)

                game = Game()

                while not game.done:
                    dir = self.calculate(game)
                    game.update(dir)

                if game.score > bestScore:
                    bestScore = game.score
                if game.score < worstScore:
                    worstScore = game.score
            
                meanScore += game.score

                if True in (game.board >= winBoard):
                    numWins += 1

            meanScore /= numAttempts

            if printInfo:
                print(f"\tResults after {numAttempts} attempts:\n")
                print(f"\tBest Score: {bestScore}")
                print(f"\tWorst Score: {worstScore}")
                print(f"\tAverage Score: {meanScore}")
                print(f"\tNumber of wins(attempts that had a tile of value 2048 or greater): {numWins}\n")

        return meanScore

    def main(self, args):
        numAttempts = 1
        if len(args) > 1:
            numAttempts = int(args[1])

        numTrials = 1
        if len(args) > 2:
            numTrials = int(args[2])

        self.test(numAttempts, numTrials, True)

if __name__ == "__main__":
    BaseAgent().main(sys.argv)
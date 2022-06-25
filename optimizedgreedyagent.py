import sys
import numpy as np
from baseagent import BaseAgent

# Agent that plays 2048 greedily by trying to maximize the score each move
class OptimizedGreedyAgent(BaseAgent):
    # determines the best direction (the String "up", "down", "left", or "right") to maximize the score for the given board state
    # in event of a tie, one of the best directions is randomly picked
    def calculate(self, game):
        originalScore = game.score
        maxScore = game.score
        bestDir = ""

        board = np.copy(game.board)

        for i in range(0,4):
            game.update(self.dirs[i])
            
            if game.score >= maxScore and not np.array_equal(game.board, board):
                maxScore = game.score
                bestDir = self.dirs[i]

            game.board = np.copy(board)
            game.score = originalScore

        if bestDir == "":
            print("wtf")
            exit()

        return bestDir

if __name__ == "__main__":
    OptimizedGreedyAgent().main(sys.argv)
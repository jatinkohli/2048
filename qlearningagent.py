import sys
import time
import random
from baseagent import BaseAgent
from greedyagent import GreedyAgent
from game import Game
import numpy as np

# Agent that plays 2048 by learning the best action for a state using Q-Learning
class QLearningAgent(BaseAgent):

    def __init__(self):
        self.prevAction = ""
        self.prevState = []
        super().__init__()

    def train(self, seconds, stepSize, futureDiscount, epsilon, file, overwrite):
        if overwrite:
            self.q_sa = {}
        else:
            self.readModel(file)

        g = GreedyAgent()
        start_time = time.time()

        printerval = seconds / 10
        while time.time() - start_time < seconds:
            # if (time.time() - start_time < printerval):
            #     print(f"{printerval / seconds}% done with training")
            #     printerval = (printerval * 10 + 1) / 10

            game = Game()

            s = game.board.flatten()
            a = g.calculate(game)
            score = game.score

            oldBoard = []

            while not game.done:
                explore = random.random() < epsilon

                oldBoard = np.copy(game.board)

                game.update(a)

                sPrime = game.board.flatten()
                aPrime = g.calculate(game)

                reward = game.score
                if np.array_equal(oldBoard, game.board):
                    reward = -game.score

                keys = self.q_sa.keys()
                if (*s, self.dirs.index(a)) not in keys:
                    self.q_sa[(*s, self.dirs.index(a))] = 0

                if (*sPrime, self.dirs.index(aPrime)) not in keys:
                    val = 0
                    if game.done:
                        val = game.score
                    self.q_sa[(*sPrime, self.dirs.index(aPrime))] = val

                # Q(s,a) += step_size * (reward for s + future_discount * max(Q(s',a')) - Q(s,a))
                self.q_sa[(*s, self.dirs.index(a))] += stepSize * (reward + futureDiscount * self.q_sa[(*sPrime, self.dirs.index(aPrime))] - self.q_sa[(*s, self.dirs.index(a))])

                if explore:
                    game.board = oldBoard
                    game.update(np.random.choice(self.dirs))
                    sPrime = game.board.flatten()
                    aPrime = g.calculate(game)

                s = sPrime
                a = aPrime
                score = game.score

        f = open(file, "w")

        for key in self.q_sa:
            line = "["

            line += "".join(str(key)[1:-1].split(","))
            line = line[:-2] + "]" + line[-2:] + " " + str(self.q_sa[key]) + "\n"

            f.write(line)

    def readModel(self, file):
        self.q_sa = {}

        with open(file) as f:
            for line in f.readlines():
                index = line.index("]")
                state = [int(s) for s in line[0:index].split(" ")]
                action = int(line[index + 2])
                q = float(line[(index + 4):len(line)])

                self.q_sa[(*state, action)] = q

    # determines the best direction (the String "up", "down", "left", or "right")
    # Uses a policy from a previously-trained Q-Learning model
    def calculate(self, game):
        bestDirs = []

        state = game.board.flatten()

        max = -1E7
        for action in range (0, 4):
            reward = 0
            if (*state, action) in self.q_sa.keys():
                reward = self.q_sa[(*state, action)]

            if reward > max:
                max = reward
                bestDirs = [self.dirs[action]]
            elif reward == max:
                bestDirs.append(self.dirs[action])

        dir = np.random.choice(bestDirs)

        if len(bestDirs) == 1 and np.array_equal(state, self.prevState) and self.prevDir == bestDirs[0]:
            print("uh oh")

        print(np.array_equal(state, self.prevState))

        self.prevDir = dir
        self.prevState = np.copy(state)

        return dir

if __name__ == "__main__":
    q = QLearningAgent()

    minutes = 60
    for i in range(int(sys.argv[2])):
        q.train(60 * minutes, 0.5, 0.7, 0.1, "qlearningmodel.txt", False)
        print(f"done with hour {(minutes / 60.0) * (i + 1)}")
            
    print("Training done, loading model...")
    q.readModel("qlearningmodel.txt")
    print(f"Results after {int(sys.argv[2]) * minutes} minutes:")
    q.main(sys.argv[:2])

    # q.readModel("qlearningmodel.txt")
    # q.main(sys.argv)
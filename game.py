import numpy as np
import random

def randomNewNumber():
    if random.randint(0, 9) == 9:
        return 4
    return 2

class Game:
    def __init__(self, file = ""):
        self.board = np.zeros((4,4), dtype=np.int16)
        self.score = 0
        self.done = False
        # self.file = open(file, "w");

        for i in range(0,2):
            self.board[self.randomPosition()] = randomNewNumber()

        # self.file.write(np.array2string(self.board, separator=','))
        # self.file.write("\n")
        # self.file.flush()

        input = self.board.reshape(1, 16)
        # print(self.model(input).numpy)

    # random empty position
    # returns tuple of (row, col)
    def randomPosition(self):
        done = False

        row = 0
        col = 0

        while not done:
            row = random.randint(0, 3)
            col = random.randint(0, 3)

            done = self.board[row][col] == 0
        
        return (row, col)

    def update(self, direction):
        oldBoard = np.copy(self.board)

        dirArray = [0, 0, 0, 0]

        if direction == "up":
            dirArray[0] = 1;

            for col in range(0, 4):
                combined = [0, 0, 0, 0] # for making sure combinations do not chain

                # start by moving everything as high as possible
                for row in range(1, 4):
                    if self.board[row][col] != 0:
                        newRow = row
                        while newRow > 0 and self.board[newRow - 1][col] == 0:
                            self.board[newRow - 1][col] = self.board[newRow][col]
                            self.board[newRow][col] = 0
                            newRow -= 1

                        # handle combinations
                        if newRow > 0 and self.board[newRow][col] == self.board[newRow - 1][col] and combined[newRow - 1] == 0:
                            self.board[newRow - 1][col] *= 2
                            self.board[newRow][col] = 0
                            
                            combined[newRow - 1] = 1
                            self.score += self.board[newRow - 1][col]

        elif direction == "left":
            dirArray[1] = 1;

            for row in range(0, 4):
                combined = [0, 0, 0, 0] # for making sure combinations do not chain

                # start by moving everything as left as possible
                for col in range(1, 4):
                    if self.board[row][col] != 0:
                        newCol = col
                        while newCol > 0 and self.board[row][newCol - 1] == 0:
                            self.board[row][newCol - 1] = self.board[row][newCol]
                            self.board[row][newCol] = 0
                            newCol -= 1

                        # handle combinations
                        if newCol > 0 and self.board[row][newCol] == self.board[row][newCol - 1] and combined[newCol - 1] == 0:
                            self.board[row][newCol - 1] *= 2
                            self.board[row][newCol] = 0

                            combined[newCol - 1] = 1
                            self.score += self.board[row][newCol - 1]

        elif direction == "down":
            dirArray[2] = 1;

            for col in range(0, 4):
                combined = [0, 0, 0, 0] # for making sure combinations do not chain

                # start by moving everything as low as possible
                for row in range(2, -1, -1):
                    if self.board[row][col] != 0:
                        newRow = row
                        while newRow < 3 and self.board[newRow + 1][col] == 0:
                            self.board[newRow + 1][col] = self.board[newRow][col]
                            self.board[newRow][col] = 0
                            newRow += 1

                        # handle combinations
                        if newRow < 3 and self.board[newRow][col] == self.board[newRow + 1][col] and combined[newRow + 1] == 0:
                            self.board[newRow + 1][col] *= 2
                            self.board[newRow][col] = 0

                            combined[newRow + 1] = 1
                            self.score += self.board[newRow + 1][col]

        elif direction == "right":
            dirArray[3] = 1;

            for row in range(0, 4):
                combined = [0, 0, 0, 0] # for making sure combinations do not chain

                # start by moving everything as right as possible
                for col in range(2, -1, -1):
                    if self.board[row][col] != 0:
                        newCol = col
                        while newCol < 3 and self.board[row][newCol + 1] == 0:
                            self.board[row][newCol + 1] = self.board[row][newCol]
                            self.board[row][newCol] = 0
                            newCol += 1

                        # handle combinations
                        if newCol < 3 and self.board[row][newCol] == self.board[row][newCol + 1] and combined[newCol + 1] == 0:
                            self.board[row][newCol + 1] *= 2
                            self.board[row][newCol] = 0

                            combined[newCol + 1] = 1
                            self.score += self.board[row][newCol + 1]

        if not np.array_equal(oldBoard, self.board):
            self.board[self.randomPosition()] = randomNewNumber()

            # print to file for model training
            # self.file.write(str(dirArray))
            # self.file.write("\n\n")
            # self.file.write(np.array2string(self.board, separator=','))
            # self.file.write("\n")
            # self.file.flush()

            # input = self.board.reshape(1, self.board.shape[0], self.board.shape[1])
            # input = self.board.reshape(1, 16)
            # print(self.model(input).numpy)

            done = True

            for row in range(0, 4):
                if done:
                    for col in range(0, 4):
                        if done and self.board[row][col] == 0:
                            done = False

                        if done and col > 0 and self.board[row][col - 1] == self.board[row][col]:
                            done = False

                        if done and col < 3 and self.board[row][col + 1] == self.board[row][col]:
                            done = False

                        if done and row > 0 and self.board[row - 1][col] == self.board[row][col]:
                            done = False

                        if done and row < 3 and self.board[row + 1][col] == self.board[row][col]:
                            done = False

            self.done = done
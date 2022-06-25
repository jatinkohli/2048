# import the pygame module, so you can use it
import pygame
from game import Game
from pygame.constants import KEYDOWN, K_a, K_d, K_s, K_w
from os.path import exists

BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

def createBackground(screen):
    screen.fill(GRAY)

    for rowBorder in range(0, 5):
        pygame.draw.rect(screen, BLACK, (0, (480 / 4) * rowBorder, 480, 5))    
    pygame.draw.rect(screen, BLACK, (0, 475, 480, 5))

    for colBorder in range(0, 5):
        pygame.draw.rect(screen, BLACK, ((480 / 4) * colBorder, 0, 5, 480))
    pygame.draw.rect(screen, BLACK, (475, 0, 5, 480))

def renderBoard(screen, board):
    createBackground(screen)

    font = pygame.font.SysFont("calibri", 50, bold=True)

    for row in range(0, 4):
        for col in range(0, 4):
            if board[row][col] != 0:
                val = str(int(board[row][col]))
                offset = (5 - len(val)) * 10 + 3

                img = font.render(val, True, BLACK)
                screen.blit(img, (offset + col * 120, 40 + row * 120))

    pygame.display.flip()

# define a main function
def main():
    pygame.init()
    pygame.display.set_caption("2048")

    # # generate file name
    # id = 0;
    # while exists("games\\game{}.txt".format(id)):
    #     id += 1;

    # game = Game("games\\game{}.txt".format(id))
    game = Game()
     
    screen = pygame.display.set_mode((480, 480)) 
    renderBoard(screen, game.board)

    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_w:
                    game.update("up")
                elif event.key == K_a:
                    game.update("left")
                elif event.key == K_s:
                    game.update("down")
                elif event.key == K_d:
                    game.update("right")

                renderBoard(screen, game.board)

            if game.done:
                print("done")
            
            if event.type == pygame.QUIT:
                print(game.score)
                exit()
     
if __name__=="__main__":
    main()
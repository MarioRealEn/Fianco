import pygame
import sys
import numpy as np
import game


def main():
    fianco = game.FiancoGame()
    while True:
        fianco.clock.tick(60)
        fianco.draw_board()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                fianco.handle_click(pygame.mouse.get_pos())

if __name__ == "__main__":
    main()
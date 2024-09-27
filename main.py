import pygame
import sys
import numpy as np
import game
from controller import Controller


def main():
    # initial_board_state = np.array([
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 1, 0, 1, 1],
    #         [1, 1, 0, 1, 0, 0, 1, 1, 1],
    #         [1, -1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, -1, 0, 0, 1, 0, 0, 0],
    #         [0, 0, -1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, -1, 0, -1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, -1, 0],
    #         [0, -1, 0, -1, 0, -1, -1, -1, -1],
    #         ], dtype=np.int8)
    fianco = game.FiancoGame()
    # Example: Set Black player to be controlled by AI
    fianco.player_types[1] = 'ai'  # Black player is AI
    # Initialize controllers for AI players
    for player, p_type in fianco.player_types.items():
        if p_type == 'ai':
            fianco.controllers[player] = Controller(player, fianco)
    fianco.run_game()

if __name__ == "__main__":
    main()
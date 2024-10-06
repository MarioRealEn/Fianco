import pygame
import sys
import numpy as np
import game
from controller import AIController


def main():
    board = np.array([
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 0, 0, 1, 0, 0, 0, 0, 1],
[-1, 1, 1, 0, 0, 1, 0, 1, 1],
[0, -1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, -1, 0, 0, 0, 0, 0, -1],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, -1, -1, 0, -1, -1],
], dtype=np.int8)
    fianco = game.FiancoGame()
    # Example: Set Black player to be controlled by AI
    fianco.player_types[1] = 'ai7'
    fianco.player_types[-1] = 'ai7' 
    # Initialize controllers for AI players
    for player, p_type in fianco.player_types.items():
        print(p_type[0:2], p_type[2])
        if p_type[0:2] == 'ai':
            fianco.controllers[player] = AIController(player, fianco, int(p_type[2]))
            print(f"Player {player} is AI with depth {p_type[2]}")
    fianco.run_game()

if __name__ == "__main__":
    main()
import pygame
import sys
import numpy as np
import game
from controller import Controller


def main():
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
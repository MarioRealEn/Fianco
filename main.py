import pygame 
import sys
import numpy as np
import game
from controller import AIController


def main():

    fianco = game.FiancoGame()

    # Show the setup page
    fianco.run_setup_menu()
    fianco.apply_setup()

    fianco.run_game()

if __name__ == "__main__":
    main()
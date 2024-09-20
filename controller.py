# import game
import numpy as np
class Controller:
    def __init__(self, player):
        self.player = player  # -1 for White, 1 for Black

    def get_move(self, board_state):
        # Placeholder for Rust function call
        # In the future, this method will call the Rust function to get the AI's move
        arr = np.array([[0,0,1,0], [1, 1, 2, 1]])
        random_index = np.random.randint(0, arr.shape[0], size=1)
        print(arr[random_index][0])
        return arr[random_index][0]
        # return np.random.choice(game.get_all_possible_moves(self.player))
        # raise NotImplementedError("AI move function is not implemented yet.")

    
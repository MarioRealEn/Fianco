import numpy as np
from fianco_brain import get_best_move  # Import the Rust AI function

class Controller:
    def __init__(self, player, game):
        self.player = player  # -1 for White, 1 for Black
        self.game = game

    def get_move(self, board_state):
        # Ensure the board_state is a NumPy array of type int8
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state, dtype=np.int8)
        else:
            board_state = board_state.astype(np.int8)

        player = self.player
        depth = 6  # Adjust search depth as needed

        try:
            best_score, from_row, from_col, to_row, to_col = get_best_move(board_state, player, depth)
            return from_row, from_col, to_row, to_col
        except ValueError:
            self.game.export_position()
            raise NotImplementedError("AI has no valid moves.")
        
    def get_move_no_ai(self, board_state):
        # Placeholder for Rust function call
        # In the future, this method will call the Rust function to get the AI's move
        arr = np.array([[0,0,1,0], [1, 1, 2, 1]])
        random_index = np.random.randint(0, arr.shape[0], size=1)
        print(arr[random_index][0])
        return arr[random_index][0]
        # return np.random.choice(game.get_all_possible_moves(self.player))
        # raise NotImplementedError("AI move function is not implemented yet.")


    
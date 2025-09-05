import numpy as np
from fianco_brain import FiancoAI  # Import the Rust AI function

class AIController:
    def __init__(self, player, game, depth=20, time=60): # Depth search will stop after 60 seconds, and time search will stop after depth 20 is reached.
        self.player = player  # -1 for White, 1 for Black
        self.game = game
        self.depth = depth
        self.time = time
        self.ai = FiancoAI(player)

    def get_move(self, board_state):
        # Ensure the board_state is a NumPy array of type int8
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state, dtype=np.int8)
        else:
            board_state = board_state.astype(np.int8)

        player = self.player
        depth = self.depth  # Adjust search depth as needed

        try:
            pv = self.ai.get_best_move(board_state, player, depth, self.time)
            best_score = pv[0]
            from_row, from_col, to_row, to_col = pv[1][0] 
            print(f"Current eval: {best_score}")
            return from_row, from_col, to_row, to_col
        except ValueError:
            self.game.export_position()
            raise NotImplementedError("AI has no valid moves.")
        


class ExportController:
    def __init__(self, player, game, export_file="fianco_export.txt"):
        self.player = player.lower()  # 'white' or 'black'
        self.game = game
        self.export_file = export_file
        self.moves = []  # List to store the moves for the player
        self.current_move_index = 0  # To keep track of the next move to play

        self.parse_export_file()

    def parse_export_file(self):
        with open(self.export_file, 'r') as f:
            lines = f.readlines()

        # Variables to track which section we're in
        parsing_white = False
        parsing_black = False

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            if line.startswith('White Moves:'):
                parsing_white = True
                parsing_black = False
                continue
            elif line.startswith('Black Moves:'):
                parsing_black = True
                parsing_white = False
                continue
            elif line.startswith('Board State:'):
                parsing_white = False
                parsing_black = False
                continue

            if parsing_white and self.player == 'white':
                self.moves.append(line)
            elif parsing_black and self.player == 'black':
                self.moves.append(line)


    def get_move(self):
        if self.current_move_index >= len(self.moves):
            raise Exception("No more moves available in the export file for player '{}'.".format(self.player))

        move_notation = self.moves[self.current_move_index]
        self.current_move_index += 1

        # Parse the move notation (e.g., 'D4->D5')
        if '->' not in move_notation:
            raise ValueError("Invalid move notation: '{}'".format(move_notation))

        from_notation, to_notation = move_notation.split('->')
        from_row, from_col = self.game.notation_to_coord(from_notation)
        to_row, to_col = self.game.notation_to_coord(to_notation)

        # Validate the move using the game's rules
        if not self.game.is_valid_move(self.player, from_row, from_col, to_row, to_col):
            raise ValueError("Invalid move according to the game rules: '{}'".format(move_notation))

        return from_row, from_col, to_row, to_col

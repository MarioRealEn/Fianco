import pygame
import numpy as np
import sys
from controller import AIController  # Import the Controller class

# Constants
ROWS, COLS = 9, 9
SQUARE_SIZE = 60  # Size of each square in pixels
MARGIN = 50       # Margin size for labels
MOVE_PANEL_WIDTH = 200  # Width of the move history panel
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 30
WIDTH = SQUARE_SIZE * COLS + MARGIN * 2 + MOVE_PANEL_WIDTH
HEIGHT = SQUARE_SIZE * ROWS + MARGIN * 2

# Colors
LIGHT_SQUARE_COLOR = (240, 217, 181)
DARK_SQUARE_COLOR = (181, 136, 99)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
TEXT_COLOR = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 255, 0)
VALID_MOVE_COLOR = (34, 139, 34)
MOVE_PANEL_BG = (245, 245, 245)
SELECTED_PIECE_COLOR = (255, 215, 0)
MARGIN_COLOR = (150, 150, 150)  # White color for margin

class FiancoGame:
    def __init__(self, 
                 board = np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 0,  1,  0,  0,  0,  0,  0,  1,  0],
            [ 0,  0,  1,  0,  0,  0,  1,  0,  0],
            [ 0,  0,  0,  1,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0, -1,  0, -1,  0,  0,  0],
            [ 0,  0, -1,  0,  0,  0, -1,  0,  0],
            [ 0, -1,  0,  0,  0,  0,  0, -1,  0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int8)):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Fianco Game')
        self.clock = pygame.time.Clock()
        self.initial_board_state = board
        self.board_state = self.initial_board_state.copy()
        self.current_player = -1 # White: -1, Black: 1
        self.selected_piece = None
        self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.white_moves = []
        self.black_moves = []
        self.game_over = False
        self.font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.undo_stack = []
        self.redo_stack = []
        self.paused = False

        # Player types: 'human' or 'ai'
        self.player_types = {
            -1: 'human',  # White player
            1: 'human'    # Black player
        }

        # Controllers for AI players
        self.controllers = {
            -1: None,
            1: None
        }

        # Buttons
        button_x = MARGIN * 2 + COLS * SQUARE_SIZE + (MOVE_PANEL_WIDTH - BUTTON_WIDTH * 2 - 10) // 2
        button_y = MARGIN

        self.undo_button_rect = pygame.Rect(
            button_x,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT
        )
        self.redo_button_rect = pygame.Rect(
            button_x + BUTTON_WIDTH + 10,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT
        )
        self.reset_button_rect = pygame.Rect(
            button_x,
            button_y + BUTTON_HEIGHT + 10,
            BUTTON_WIDTH,
            BUTTON_HEIGHT
        )
        self.export_button_rect = pygame.Rect(
            button_x + BUTTON_WIDTH + 10,
            button_y + BUTTON_HEIGHT + 10,
            BUTTON_WIDTH,
            BUTTON_HEIGHT
        )
        # Add play/pause button
        self.play_button_rect = pygame.Rect(
            button_x,
            button_y + (BUTTON_HEIGHT + 10) * 2,  # Position it below existing buttons
            BUTTON_WIDTH * 2 + 10,  # Span two columns
            BUTTON_HEIGHT
        )

    def draw_board(self):
        # Fill background with margin color
        self.screen.fill(MARGIN_COLOR)
        # Draw the board with a checkered pattern
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(
                    MARGIN + col * SQUARE_SIZE,
                    MARGIN + row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
                if (row + col) % 2 == 0:
                    color = LIGHT_SQUARE_COLOR
                else:
                    color = DARK_SQUARE_COLOR
                pygame.draw.rect(self.screen, color, rect)

        # Highlight selected piece
        if self.selected_piece:
            row, col = self.selected_piece
            highlight_rect = pygame.Rect(
                MARGIN + col * SQUARE_SIZE,
                MARGIN + row * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE
            )
            pygame.draw.rect(self.screen, SELECTED_PIECE_COLOR, highlight_rect, 4)
        # Highlight valid moves
        for move in self.valid_moves:
            _, _, row, col = move
            row = int(row)
            col = int(col)
            center = (
                MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
            )
            pygame.draw.circle(self.screen, VALID_MOVE_COLOR, center, 10)
            # print(row, col, center)
        # Draw pieces
        player_positions = np.argwhere(self.board_state != 0)
        for row, col in player_positions:
            piece = self.board_state[row, col]
            if piece == 1:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            pygame.draw.circle(
                self.screen, color,
                (
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                ),
                SQUARE_SIZE // 2 - 10
            )
        # Draw coordinates
        self.draw_coordinates()
        # Draw move history
        self.draw_move_history()
        # Draw buttons
        self.draw_buttons()
        pygame.display.flip()

    def draw_coordinates(self):
        # Draw column labels (A-I) at the bottom only
        for col in range(COLS):
            label = self.font.render(chr(ord('A') + col), True, TEXT_COLOR)
            label_rect = label.get_rect(
                center=(
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    HEIGHT - MARGIN // 2
                )
            )
            self.screen.blit(label, label_rect)
        # Draw row labels (1-9) on the left side only
        for row in range(ROWS):
            label = self.font.render(str(ROWS - row), True, TEXT_COLOR)
            label_rect = label.get_rect(
                center=(
                    MARGIN // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                )
            )
            self.screen.blit(label, label_rect)

    def draw_move_history(self):
        # Background for the move history panel
        panel_rect = pygame.Rect(
            MARGIN * 2 + COLS * SQUARE_SIZE,
            0,
            MOVE_PANEL_WIDTH,
            HEIGHT
        )
        pygame.draw.rect(self.screen, MOVE_PANEL_BG, panel_rect)
        # Column positions
        col1_x = MARGIN * 2 + COLS * SQUARE_SIZE + 10
        col2_x = col1_x + MOVE_PANEL_WIDTH // 2 - 10
        start_y = MARGIN + (BUTTON_HEIGHT + 10) * 3 + 20
        line_height = 20
        # Calculate how many moves can be displayed
        max_displayed_moves = (HEIGHT - start_y - line_height) // line_height
        max_moves = max(len(self.white_moves), len(self.black_moves))
        # Determine the range of moves to display
        start_index = max(0, max_moves - max_displayed_moves)
        # Draw column headers
        header_white = self.font.render("White", True, TEXT_COLOR)
        header_black = self.font.render("Black", True, TEXT_COLOR)
        self.screen.blit(header_white, (col1_x, start_y - line_height))
        self.screen.blit(header_black, (col2_x, start_y - line_height))
        for idx in range(start_index, max_moves):
            i = idx - start_index  # Adjusted index for display
            if idx < len(self.white_moves):
                move_text = self.font.render(self.white_moves[idx], True, TEXT_COLOR)
                self.screen.blit(move_text, (col1_x, start_y + i * line_height))
            if idx < len(self.black_moves):
                move_text = self.font.render(self.black_moves[idx], True, TEXT_COLOR)
                self.screen.blit(move_text, (col2_x, start_y + i * line_height))

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        # Draw undo button
        self.draw_button(self.undo_button_rect, 'Undo', mouse_pos)
        # Draw redo button
        self.draw_button(self.redo_button_rect, 'Redo', mouse_pos)
        # Draw reset button
        self.draw_button(self.reset_button_rect, 'Reset', mouse_pos)
        # Draw export button
        self.draw_button(self.export_button_rect, 'Export', mouse_pos)
        # Draw play/pause button
        button_text = 'Play' if self.paused else 'Pause'
        self.draw_button(self.play_button_rect, button_text, mouse_pos)

    def draw_button(self, rect, text, mouse_pos):
        if rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR
        # Draw rounded rectangle
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        button_text = self.font.render(text, True, TEXT_COLOR)
        button_text_rect = button_text.get_rect(center=rect.center)
        self.screen.blit(button_text, button_text_rect)

    def coord_to_notation(self, row, col):
        col_label = chr(ord('A') + col)
        row_label = str(ROWS - row)
        return col_label + row_label
    
    def notation_to_coord(self, notation):
        col_label = notation[0].upper()
        row_label = notation[1:]
        col = ord(col_label) - ord('A')
        row = ROWS - int(row_label)
        return row, col

    def get_player_label(self, player):
        return 'White' if player == -1 else 'Black'

    def get_possible_captures(self, player):
        captures = []
        direction = 1 if player == 1 else -1
        positions = np.argwhere(self.board_state == player)
        for i, j in positions:
            delta_rows = 2 * direction
            enemy_rows = i + delta_rows // 2
            enemy_cols = j + np.array([-1, 1])
            land_rows = i + delta_rows
            land_cols = j + np.array([-2, 2])
            for k in range(2):
                enemy_row = enemy_rows
                enemy_col = enemy_cols[k]
                land_row = land_rows
                land_col = land_cols[k]
                if 0 <= enemy_row < ROWS and 0 <= enemy_col < COLS and \
                   0 <= land_row < ROWS and 0 <= land_col < COLS:
                    if self.board_state[enemy_row, enemy_col] == -player and \
                       self.board_state[land_row, land_col] == 0:
                        captures.append([i, j, land_row, land_col])
        return np.array(captures, dtype=np.int8)

    def get_all_possible_moves(self, player):
        moves = []
        direction = 1 if player == 1 else -1
        positions = np.argwhere(self.board_state == player)
        # Move forward
        forward_positions = positions + np.array([direction, 0])
        valid_mask = (forward_positions[:, 0] >= 0) & (forward_positions[:, 0] < ROWS)
        forward_positions = forward_positions[valid_mask]
        positions_fwd = positions[valid_mask]
        empty_mask = self.board_state[forward_positions[:, 0], forward_positions[:, 1]] == 0
        for pos, fwd_pos in zip(positions_fwd[empty_mask], forward_positions[empty_mask]):
            moves.append([pos[0], pos[1], fwd_pos[0], fwd_pos[1]])
        # Move sideways
        for delta_col in [-1, 1]:
            side_positions = positions + np.array([0, delta_col])
            valid_mask = (side_positions[:, 1] >= 0) & (side_positions[:, 1] < COLS)
            side_positions = side_positions[valid_mask]
            positions_side = positions[valid_mask]
            empty_mask = self.board_state[side_positions[:, 0], side_positions[:, 1]] == 0
            for pos, side_pos in zip(positions_side[empty_mask], side_positions[empty_mask]):
                moves.append([pos[0], pos[1], side_pos[0], side_pos[1]])
        return np.array(moves, dtype=np.int8)

    def get_valid_moves(self, player):
        captures = self.get_possible_captures(player)
        if captures.size > 0:
            return captures
        else:
            moves = self.get_all_possible_moves(player)
            return moves

    def make_move(self, from_row, from_col, to_row, to_col):
        # Save current state for undo
        self.undo_stack.append((
            self.board_state.copy(),
            self.current_player,
            self.white_moves.copy(),
            self.black_moves.copy()
        ))
        # Clear redo stack
        self.redo_stack.clear()
        player = self.current_player
        self.board_state[from_row, from_col] = 0
        self.board_state[to_row, to_col] = player
        # Record the move without player information
        from_notation = self.coord_to_notation(from_row, from_col)
        to_notation = self.coord_to_notation(to_row, to_col)
        move_str = f"{from_notation}->{to_notation}"
        if player == -1:
            self.white_moves.append(move_str)
        else:
            self.black_moves.append(move_str)
        if abs(from_row - to_row) == 2:
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            self.board_state[captured_row, captured_col] = 0
        self.draw_board()

    def undo_move(self):
        if self.undo_stack:
            # Save current state for redo
            self.redo_stack.append((
                self.board_state.copy(),
                self.current_player,
                self.white_moves.copy(),
                self.black_moves.copy()
            ))
            # Restore the previous state
            self.board_state, self.current_player, self.white_moves, self.black_moves = self.undo_stack.pop()
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
            self.game_over = False
            self.paused = True  # Pause the game after undo
            self.draw_board()

    def redo_move(self):
        if self.redo_stack:
            # Save current state for undo
            self.undo_stack.append((
                self.board_state.copy(),
                self.current_player,
                self.white_moves.copy(),
                self.black_moves.copy()
            ))
            # Restore the next state
            self.board_state, self.current_player, self.white_moves, self.black_moves = self.redo_stack.pop()
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
            self.game_over = False
            self.paused = True  # Pause the game after redo
            self.draw_board()


    def check_for_win(self):
        player = self.current_player
        target_row = 8 if player == 1 else 0
        if player in self.board_state[target_row]:
            self.draw_board()
            self.game_over = True
            text = self.large_font.render(f'{self.get_player_label(player)} Wins!', True, TEXT_COLOR)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(text, text_rect)
            self.export_position()
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()

    def select_piece(self, row, col):
        if self.board_state[row, col] == self.current_player:
            self.selected_piece = (row, col)
            all_moves = self.get_valid_moves(self.current_player)
            # Filter moves for the selected piece
            self.valid_moves = all_moves[np.all(all_moves[:, 0:2] == [row, col], axis=1)]
        else:
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.draw_board()

    def handle_click(self, pos):
        if self.game_over:
            return
        x, y = pos
        # Check if click is on play/pause button
        if self.play_button_rect.collidepoint(x, y):
            self.paused = not self.paused  # Toggle paused state
            self.draw_board()  # Redraw to update button label
            return
        # Check if click is on undo button
        if self.undo_button_rect.collidepoint(x, y):
            self.undo_move()
            return
        # Check if click is on redo button
        if self.redo_button_rect.collidepoint(x, y):
            self.redo_move()
            return
        # Check if click is on reset button
        if self.reset_button_rect.collidepoint(x, y):
            self.reset_game()
            return
        # Check if click is on export button
        if self.export_button_rect.collidepoint(x, y):
            self.export_position()
            return
        # If it's AI's turn, ignore clicks
        if self.player_types[self.current_player][0:2] == 'ai':
            return
        col = (x - MARGIN) // SQUARE_SIZE
        row = (y - MARGIN) // SQUARE_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            if self.selected_piece:
                move_indices = np.where((self.valid_moves[:, 2] == row) & (self.valid_moves[:, 3] == col))[0]
                if move_indices.size > 0:
                    # move = self.valid_moves[move_indices[0]]
                    from_row, from_col = self.selected_piece
                    self.selected_piece = None #This goes before the make_move because make_move draws the board
                    self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4) #Same for this
                    self.make_move(from_row, from_col, row, col)
                    self.check_for_win()
                    self.current_player *= -1
                    return
                else:
                    self.select_piece(row, col)
            else:
                self.select_piece(row, col)
        else:
            # Clicked outside the board
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)

    def reset_game(self):
        # Reset the board state
        self.board_state = self.initial_board_state.copy()
        self.current_player = -1
        self.selected_piece = None
        self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.white_moves = []
        self.black_moves = []
        self.undo_stack = []
        self.redo_stack = []
        self.game_over = False
        self.draw_board()

    def export_position(self):
        # Export the current position and move history to a text file
        with open('fianco_export.txt', 'w') as f:
            f.write('Board State:\n')
            f.write('board = np.array([\n')
            for row in self.board_state:
                f.write('[' + ', '.join(map(str, row)) + '],\n')
            f.write('], dtype=np.int8)\n\n')
            f.write('\nWhite Moves:\n')
            for move in self.white_moves:
                f.write(move + '\n')
            f.write('\nBlack Moves:\n')
            for move in self.black_moves:
                f.write(move + '\n')
        print('Position exported to fianco_export.txt')

    def handle_ai_move(self):
        controller = self.controllers[self.current_player]
        if controller is None:
            return
        try:
            move = controller.get_move(self.board_state)
        except NotImplementedError as e:
            print(e)
            self.game_over = True
            return
        from_row, from_col, to_row, to_col = move
        # pygame.time.wait(500)  # Delay for better visualization
        self.selected_piece = None
        self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.make_move(from_row, from_col, to_row, to_col)
        self.check_for_win()
        self.current_player *= -1

    def run_game(self):
        self.draw_board()
        while True:
            self.clock.tick(60)
            if self.game_over:
                pygame.quit()
                sys.exit()
            # If it's AI's turn and the game is not paused
            if self.player_types[self.current_player][0:2] == 'ai' and not self.paused:
                self.handle_ai_move()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())

if __name__ == "__main__":
    game = FiancoGame()
    # Example: Set Black player to be controlled by AI
    game.player_types[1] = 'ai'  # Black player is AI
    # Initialize controllers for AI players
    for player, p_type in game.player_types.items():
        if p_type == 'ai':
            game.controllers[player] = AIController(player)
    game.run_game()


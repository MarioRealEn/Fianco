import pygame
import numpy as np
import sys

# Constants
ROWS, COLS = 9, 9
SQUARE_SIZE = 60  # Size of each square in pixels
MARGIN = 50       # Margin size for labels
MOVE_PANEL_WIDTH = 200  # Width of the move history panel
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 30
WIDTH = SQUARE_SIZE * COLS + MARGIN * 2 + MOVE_PANEL_WIDTH
HEIGHT = SQUARE_SIZE * ROWS + MARGIN * 2

# RGB Colors
BOARD_COLOR = (235, 190, 150)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (211, 211, 211)

class FiancoGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Fianco Game')
        self.clock = pygame.time.Clock()
        self.initial_board_state = np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 0,  1,  0,  0,  0,  0,  0,  1,  0],
            [ 0,  0,  1,  0,  0,  0,  1,  0,  0],
            [ 0,  0,  0,  1,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0, -1,  0, -1,  0,  0,  0],
            [ 0,  0, -1,  0,  0,  0, -1,  0,  0],
            [ 0, -1,  0,  0,  0,  0,  0, -1,  0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ], dtype=np.int8)
        self.board_state = self.initial_board_state.copy()
        self.current_player = -1
        self.selected_piece = None
        self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.move_history = []
        self.game_over = False
        self.font = pygame.font.SysFont(None, 24)
        self.undo_stack = []
        self.redo_stack = []

        # Buttons
        # Adjusted buttons to fit four buttons (Undo, Redo, Reset, Export)
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

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        # Draw grid
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(
                    MARGIN + col * SQUARE_SIZE,
                    MARGIN + row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        # Draw pieces using NumPy operations
        player_positions = np.argwhere(self.board_state != 0)
        for row, col in player_positions:
            piece = self.board_state[row, col]
            color = BLACK if piece == 1 else WHITE
            pygame.draw.circle(
                self.screen, color,
                (
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                ),
                SQUARE_SIZE // 2 - 10
            )
        # Highlight valid moves
        for move in self.valid_moves:
            _, _, row, col = move
            pygame.draw.circle(
                self.screen, BLACK,
                (
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                ),
                10
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
            label = self.font.render(chr(ord('A') + col), True, BLACK)
            label_rect = label.get_rect(
                center=(
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    HEIGHT - MARGIN // 2
                )
            )
            self.screen.blit(label, label_rect)
        # Draw row labels (1-9) on the left side only
        for row in range(ROWS):
            label = self.font.render(str(ROWS - row), True, BLACK)
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
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        # Draw the moves
        start_y = MARGIN + BUTTON_HEIGHT * 2 + 30
        line_height = 20
        moves_to_show = self.move_history[-(HEIGHT // line_height - 4):]  # Show last N moves
        for i, move in enumerate(moves_to_show):
            move_text = self.font.render(move, True, BLACK)
            self.screen.blit(
                move_text,
                (
                    MARGIN * 2 + COLS * SQUARE_SIZE + 10,
                    start_y + i * line_height
                )
            )

    def draw_buttons(self):
        # Draw undo button
        pygame.draw.rect(self.screen, GRAY, self.undo_button_rect)
        undo_text = self.font.render('Undo', True, BLACK)
        undo_text_rect = undo_text.get_rect(center=self.undo_button_rect.center)
        self.screen.blit(undo_text, undo_text_rect)
        # Draw redo button
        pygame.draw.rect(self.screen, GRAY, self.redo_button_rect)
        redo_text = self.font.render('Redo', True, BLACK)
        redo_text_rect = redo_text.get_rect(center=self.redo_button_rect.center)
        self.screen.blit(redo_text, redo_text_rect)
        # Draw reset button
        pygame.draw.rect(self.screen, GRAY, self.reset_button_rect)
        reset_text = self.font.render('Reset', True, BLACK)
        reset_text_rect = reset_text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(reset_text, reset_text_rect)
        # Draw export button
        pygame.draw.rect(self.screen, GRAY, self.export_button_rect)
        export_text = self.font.render('Export', True, BLACK)
        export_text_rect = export_text.get_rect(center=self.export_button_rect.center)
        self.screen.blit(export_text, export_text_rect)

    def coord_to_notation(self, row, col):
        col_label = chr(ord('A') + col)
        row_label = str(ROWS - row)
        return col_label + row_label

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
            return captures, True
        else:
            moves = self.get_all_possible_moves(player)
            return moves, False

    def make_move(self, from_row, from_col, to_row, to_col):
        # Save current state for undo
        self.undo_stack.append((self.board_state.copy(), self.current_player, self.move_history.copy()))
        # Clear redo stack
        self.redo_stack.clear()
        player = self.current_player
        self.board_state[from_row, from_col] = 0
        self.board_state[to_row, to_col] = player
        # Record the move with player information
        from_notation = self.coord_to_notation(from_row, from_col)
        to_notation = self.coord_to_notation(to_row, to_col)
        player_label = self.get_player_label(player)
        move_str = f"{player_label}: {from_notation}->{to_notation}"
        self.move_history.append(move_str)
        if abs(from_row - to_row) == 2:
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            self.board_state[captured_row, captured_col] = 0

    def undo_move(self):
        if self.undo_stack:
            # Save current state for redo
            self.redo_stack.append((self.board_state.copy(), self.current_player, self.move_history.copy()))
            # Restore the previous state
            self.board_state, self.current_player, self.move_history = self.undo_stack.pop()
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
            self.game_over = False

    def redo_move(self):
        if self.redo_stack:
            # Save current state for undo
            self.undo_stack.append((self.board_state.copy(), self.current_player, self.move_history.copy()))
            # Restore the next state
            self.board_state, self.current_player, self.move_history = self.redo_stack.pop()
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
            self.game_over = False

    def check_for_win(self):
        player = self.current_player
        target_row = 8 if player == 1 else 0
        if player in self.board_state[target_row]:
            self.draw_board()
            self.game_over = True
            font = pygame.font.SysFont(None, 48)
            player_label = self.get_player_label(player)
            text = font.render(f'{player_label} Player wins!', True, BLACK)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()

    def select_piece(self, row, col):
        if self.board_state[row, col] == self.current_player:
            self.selected_piece = (row, col)
            all_moves, is_capture = self.get_valid_moves(self.current_player)
            # Filter moves for the selected piece
            self.valid_moves = all_moves[np.all(all_moves[:, 0:2] == [row, col], axis=1)]
            if is_capture:
                # If capturing is mandatory, filter out non-capture moves
                self.valid_moves = self.valid_moves[np.abs(self.valid_moves[:, 0] - self.valid_moves[:, 2]) == 2]
        else:
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)

    def handle_click(self, pos):
        if self.game_over:
            return
        x, y = pos
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
        col = (x - MARGIN) // SQUARE_SIZE
        row = (y - MARGIN) // SQUARE_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            if self.selected_piece:
                move_indices = np.where((self.valid_moves[:, 2] == row) & (self.valid_moves[:, 3] == col))[0]
                if move_indices.size > 0:
                    move = self.valid_moves[move_indices[0]]
                    from_row, from_col = self.selected_piece
                    self.make_move(from_row, from_col, row, col)
                    self.selected_piece = None
                    self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
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
        self.move_history = []
        self.undo_stack = []
        self.redo_stack = []
        self.game_over = False

    def export_position(self):
        # Export the current position and move history to a text file
        with open('fianco_export.txt', 'w') as f:
            f.write('Board State:\n')
            for row in self.board_state:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\nMove History:\n')
            for move in self.move_history:
                f.write(move + '\n')
        print('Position exported to fianco_export.txt')

    def run_game(self):
        while True:
            self.clock.tick(60)
            self.draw_board()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())

if __name__ == "__main__":
    game = FiancoGame()
    game.run_game()



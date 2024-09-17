import pygame
import numpy as np
import sys

# Constants
ROWS, COLS = 9, 9
SQUARE_SIZE = 60  # Size of each square in pixels
MARGIN = 50       # Margin size for labels
WIDTH = SQUARE_SIZE * COLS + MARGIN * 2
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
        self.board_state = np.array([
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
        self.current_player = -1
        self.selected_piece = None
        self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)
        self.game_over = False
        self.font = pygame.font.SysFont(None, 24)

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
        for pos in player_positions:
            row, col = pos
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
        pygame.display.flip()

    def draw_coordinates(self):
        # Draw column labels (A-I)
        for col in range(COLS):
            label = self.font.render(chr(ord('A') + col), True, BLACK)
            label_rect = label.get_rect(
                center=(
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    MARGIN // 2
                )
            )
            self.screen.blit(label, label_rect)
            # Bottom labels
            label_rect = label.get_rect(
                center=(
                    MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    HEIGHT - MARGIN // 2
                )
            )
            self.screen.blit(label, label_rect)
        # Draw row labels (1-9)
        for row in range(ROWS):
            label = self.font.render(str(ROWS - row), True, BLACK)
            label_rect = label.get_rect(
                center=(
                    MARGIN // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                )
            )
            self.screen.blit(label, label_rect)
            # Right labels
            label_rect = label.get_rect(
                center=(
                    WIDTH - MARGIN // 2,
                    MARGIN + row * SQUARE_SIZE + SQUARE_SIZE // 2
                )
            )
            self.screen.blit(label, label_rect)

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
        player = self.current_player
        self.board_state[from_row, from_col] = 0
        self.board_state[to_row, to_col] = player
        if abs(from_row - to_row) == 2:
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            self.board_state[captured_row, captured_col] = 0

    def check_for_win(self):
        def get_player_label(player):
            return 'White' if player == -1 else 'Black'
        player = self.current_player
        target_row = 8 if player == 1 else 0
        if player in self.board_state[target_row]:
            self.draw_board()
            self.game_over = True
            font = pygame.font.SysFont(None, 48)
            text = font.render(f'{get_player_label(player)} Player wins!', True, BLACK)
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
        else:
            self.selected_piece = None
            self.valid_moves = np.array([], dtype=np.int8).reshape(0, 4)

    def handle_click(self, pos):
        if self.game_over:
            return
        x, y = pos
        col = (x - MARGIN) // SQUARE_SIZE
        row = (y - MARGIN) // SQUARE_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            if self.selected_piece:
                for move in self.valid_moves:
                    if move[2] == row and move[3] == col:
                        from_row, from_col = self.selected_piece
                        self.make_move(from_row, from_col, row, col)
                        self.selected_piece = None
                        self.valid_moves = []
                        self.check_for_win()
                        self.current_player *= -1
                        return
                self.select_piece(row, col)
            else:
                self.select_piece(row, col)
        else:
            # Clicked outside the board
            self.selected_piece = None
            self.valid_moves = []

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


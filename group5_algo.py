import pygame
import sys
import copy
import random
import time
import os  # Import os module for relaunching main.py

# --- UI Constants Definition ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
BOARD_SIZE = 450
TILE_SIZE = BOARD_SIZE // 3

GAME_TOP_MARGIN = 30
BOARD_RECT_TOP = GAME_TOP_MARGIN
BOARD_RECT_BOTTOM = BOARD_RECT_TOP + BOARD_SIZE

TILE_FONT_SIZE_CALC = TILE_SIZE // 2
MSG_FONT_SIZE_CALC = 28
BUTTON_FONT_SIZE_CALC = 18

MESSAGE_Y_CENTER = BOARD_RECT_BOTTOM + (MSG_FONT_SIZE_CALC // 2) + 15

BUTTON_HEIGHT = 40
BUTTON_ROW_SPACING = 10
BUTTONS_START_Y = MESSAGE_Y_CENTER + (MSG_FONT_SIZE_CALC // 2) + 15

INPUT_BOX_HEIGHT = 30
INPUT_LABEL_OFFSET_Y = 22
INPUT_BOXES_START_Y = BUTTONS_START_Y + (2 * BUTTON_HEIGHT) + BUTTON_ROW_SPACING + 20

MAX_ANIMATION_STEPS_NO_GOAL = 30
BACKTRACKING_MAX_DEPTH = 35
FORWARD_CHECKING_MAX_DEPTH = 35
MIN_CONFLICTS_MAX_ITERATIONS = 200

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)
TILE_BORDER_COLOR = (0, 0, 139)
GREEN = (0, 180, 0)
GREY = (128, 128, 128)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 220)
BUTTON_TEXT_COLOR = WHITE

pygame.font.init()
PREFERRED_FONT_NAME = "Arial"  # Changed from VIETNAMESE_FONT_NAME
try:
    TILE_FONT = pygame.font.SysFont(PREFERRED_FONT_NAME, TILE_FONT_SIZE_CALC)
    MSG_FONT = pygame.font.SysFont(PREFERRED_FONT_NAME, MSG_FONT_SIZE_CALC)
    BUTTON_FONT = pygame.font.SysFont(PREFERRED_FONT_NAME, BUTTON_FONT_SIZE_CALC)
    test_render = MSG_FONT.render("Sample Text", True, BLACK)  # Changed text
    if test_render.get_width() < 10:  # Basic check if font loaded something
        print(f"Warning: Font '{PREFERRED_FONT_NAME}' might not be optimal or fully loaded.")  # Changed message
        raise pygame.error("Font loading issue or poor rendering")
except pygame.error as e:
    print(f"Warning: Could not load system font '{PREFERRED_FONT_NAME}': {e}")  # Changed message
    print("Using Pygame default font. Display may not be optimal.")  # Changed message
    TILE_FONT = pygame.font.Font(None, TILE_FONT_SIZE_CALC)
    MSG_FONT = pygame.font.Font(None, MSG_FONT_SIZE_CALC)
    BUTTON_FONT = pygame.font.Font(None, BUTTON_FONT_SIZE_CALC)

Moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BOARD_MARGIN_HORIZONTAL = (SCREEN_WIDTH - BOARD_SIZE) // 2


def find_tile_pos(board, tile_value):
    for r_idx, row in enumerate(board):
        for c_idx, val in enumerate(row):
            if val == tile_value:
                return r_idx, c_idx
    return None


def are_tiles_adjacent_3_6(board):
    pos3 = find_tile_pos(board, 3)
    pos6 = find_tile_pos(board, 6)
    if not pos3 or not pos6:
        return False
    r3, c3 = pos3
    r6, c6 = pos6
    return abs(r3 - r6) + abs(c3 - c6) == 1


def Find_Empty(board):
    return find_tile_pos(board, 0)


def Check(x, y):
    return 0 <= x < 3 and 0 <= y < 3


def Chinh_Sua_Ma_Tran(board, x, y, new_x, new_y):  # Function name kept for consistency if referenced elsewhere
    new_board = copy.deepcopy(board)
    new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
    return new_board


def Rangbuoc(state):  # Function name kept
    pos_empty = Find_Empty(state)
    if pos_empty is None:
        return []
    x, y = pos_empty
    possible_deltas = []
    if x > 0:
        possible_deltas.append((-1, 0))
    if x < 2:
        possible_deltas.append((1, 0))
    if y > 0:
        possible_deltas.append((0, -1))
    if y < 2:
        possible_deltas.append((0, 1))
    return possible_deltas


def Find_X(tile_value, goal_board):  # Function name kept
    return find_tile_pos(goal_board, tile_value)


def Manhattan_Heuristic(current_board, goal_board):
    distance = 0
    for r in range(3):
        for c in range(3):
            tile_value = current_board[r][c]
            if tile_value != 0:
                goal_r_c = Find_X(tile_value, goal_board)
                if goal_r_c is not None:
                    goal_r, goal_c = goal_r_c
                    distance += abs(r - goal_r) + abs(c - goal_c)
    return distance


def DoiMotKhacNhau(board):  # Function name kept
    seen = set()
    count = 0
    if not isinstance(board, list) or len(board) != 3:
        return False
    for row in board:
        if not isinstance(row, list) or len(row) != 3:
            return False
        for val in row:
            if not isinstance(val, int) or not (0 <= val <= 8):
                return False
            if val in seen:
                return False
            seen.add(val)
            count += 1
    return count == 9


def GiaiDuoc(puzzle):  # Function name kept
    flat_puzzle = [num for row in puzzle for num in row if num != 0]
    inversions = 0
    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1
    return inversions % 2 == 0


def Backtracking(state, goal, path=None, visited=None, depth=35):
    if path is None:
        path = [state]
    if visited is None:
        visited = {tuple(map(tuple, state))}
    if state == goal:
        return path
    if depth == 0:
        return None

    empty_pos = Find_Empty(state)
    if empty_pos is None:
        return None
    X, Y = empty_pos

    possible_moves = Rangbuoc(state)
    currently_3_6_adjacent = are_tiles_adjacent_3_6(state)

    for dx, dy in possible_moves:
        new_x, new_y = X + dx, Y + dy
        new_state = Chinh_Sua_Ma_Tran(state, X, Y, new_x, new_y)

        if currently_3_6_adjacent and not are_tiles_adjacent_3_6(new_state):
            continue

        new_state_tuple = tuple(map(tuple, new_state))
        if new_state_tuple not in visited:
            visited.add(new_state_tuple)
            new_path = Backtracking(new_state, goal, path + [new_state], visited, depth - 1)
            if new_path:
                return new_path
            visited.remove(new_state_tuple)
    return None


def Forward_Check_Pruning(new_state, visited, goal):
    return tuple(map(tuple, new_state)) not in visited


def Forward_Checking(state, goal, path=None, visited=None, depth=35):
    if path is None:
        path = [state]
    if visited is None:
        visited = {tuple(map(tuple, state))}
    if state == goal:
        return path
    if depth == 0:
        return None
    empty_pos = Find_Empty(state)
    if empty_pos is None:
        return None
    X, Y = empty_pos
    possible_moves = Rangbuoc(state)
    for dx, dy in possible_moves:
        new_x, new_y = X + dx, Y + dy
        new_state = Chinh_Sua_Ma_Tran(state, X, Y, new_x, new_y)

        if Forward_Check_Pruning(new_state, visited, goal):
            new_state_tuple = tuple(map(tuple, new_state))
            visited.add(new_state_tuple)
            result_path = Forward_Checking(new_state, goal, path + [new_state], visited, depth - 1)
            if result_path:
                return result_path
            visited.remove(new_state_tuple)
    return None


def AC3_Generate_Board():
    max_attempts = 2000  # Có thể tăng số lần thử nếu cần
    for _ in range(max_attempts):
        nums = list(range(9))
        random.shuffle(nums)
        board = [nums[0:3], nums[3:6], nums[6:9]]
        if DoiMotKhacNhau(board) and GiaiDuoc(board) and are_tiles_adjacent_3_6(board):  # Thêm điều kiện 3&6 kề nhau
            return board

    # Fallback board phải thỏa mãn tất cả các điều kiện, bao gồm 3 và 6 kề nhau
    fallback_board = [[1, 2, 3], [4, 5, 6], [0, 7, 8]]
    # Kiểm tra fallback_board (quan trọng!)
    if not (DoiMotKhacNhau(fallback_board) and
            GiaiDuoc(fallback_board) and
            are_tiles_adjacent_3_6(fallback_board)):
        print("CRITICAL ERROR: Standard fallback board IS INVALID or does not meet all constraints (3&6 adjacent)!")

    print(f"Warning: Could not generate a board with 3&6 adjacent after {max_attempts} attempts. Using fallback.")
    return fallback_board


def Min_Conflicts_Search(start_state, goal_state, max_iterations=200):
    current_state = copy.deepcopy(start_state)
    path = [copy.deepcopy(current_state)]
    if current_state == goal_state:
        return path
    for _ in range(max_iterations):
        if current_state == goal_state:
            return path
        empty_pos = Find_Empty(current_state)
        if empty_pos is None:
            return path
        empty_x, empty_y = empty_pos
        possible_next_states_info = []
        valid_moves = Rangbuoc(current_state)
        if not valid_moves:
            return path
        for dx, dy in valid_moves:
            new_x, new_y = empty_x + dx, empty_y + dy
            next_s = Chinh_Sua_Ma_Tran(current_state, empty_x, empty_y, new_x, new_y)
            conflict_score = Manhattan_Heuristic(next_s, goal_state)
            possible_next_states_info.append({'state': next_s, 'score': conflict_score})
        if not possible_next_states_info:
            return path
        possible_next_states_info.sort(key=lambda x: x['score'])
        min_conflict_score = possible_next_states_info[0]['score']
        best_options = [info['state'] for info in possible_next_states_info if info['score'] == min_conflict_score]
        current_state = random.choice(best_options)
        path.append(copy.deepcopy(current_state))
    return path


def get_tile_rect(row, col):
    x = BOARD_MARGIN_HORIZONTAL + col * TILE_SIZE
    y = BOARD_RECT_TOP + row * TILE_SIZE
    return pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)


def draw_board_static(screen, board_state, highlight_coords=None):
    for r in range(3):
        for c in range(3):
            rect = get_tile_rect(r, c)
            is_highlighted = highlight_coords and r == highlight_coords[0] and c == highlight_coords[1]
            tile_color = GREEN if is_highlighted else LIGHT_BLUE
            pygame.draw.rect(screen, tile_color, rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR, rect, 3)
            tile_value = board_state[r][c]
            if tile_value != 0:
                text_surf = TILE_FONT.render(str(tile_value), True, BLACK)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)


def draw_main_message(screen, message, y_center):
    if MSG_FONT:
        text_surf = MSG_FONT.render(message, True, BLACK)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y_center))
        screen.blit(text_surf, text_rect)


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color, font, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.action = action
        self.is_hovered = False

    def draw(self, screen):
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, current_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, TILE_BORDER_COLOR, self.rect, 2, border_radius=5)
        if self.font:
            text_surf = self.font.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered and self.action:
                self.action()


def parse_board_from_string(input_str):
    try:
        nums_str = input_str.split(',')
        if len(nums_str) != 9:
            return None
        nums = [int(n.strip()) for n in nums_str]
        temp_board = [nums[i:i + 3] for i in range(0, 9, 3)]
        if not DoiMotKhacNhau(temp_board):
            return None
        return temp_board
    except:
        return None


def animate_solution_path(screen, clock, start_board, solution_path, animation_speed=0.2):
    if not solution_path or len(solution_path) < 2:
        return
    # current_display_board = copy.deepcopy(start_board) # Not strictly needed here as we rebuild display
    for step_idx in range(len(solution_path) - 1):
        board_before_move = solution_path[step_idx]
        next_board_state = solution_path[step_idx + 1]
        empty_prev_r, empty_prev_c = Find_Empty(board_before_move)
        empty_next_r, empty_next_c = Find_Empty(next_board_state)
        if empty_prev_r is None or empty_next_r is None:
            continue
        moving_tile_value = board_before_move[empty_next_r][empty_next_c]
        tile_start_rect = get_tile_rect(empty_next_r, empty_next_c)
        tile_end_rect = get_tile_rect(empty_prev_r, empty_prev_c)
        num_frames = max(1, int(animation_speed * 30))
        dx_tile = (tile_end_rect.x - tile_start_rect.x) / num_frames
        dy_tile = (tile_end_rect.y - tile_start_rect.y) / num_frames
        for frame in range(num_frames + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            screen.fill(WHITE)
            # Draw static part of the board
            for r_idx_static in range(3):
                for c_idx_static in range(3):
                    is_moving_tile_original_pos = (r_idx_static == empty_next_r and c_idx_static == empty_next_c)
                    is_blank_original_pos = (r_idx_static == empty_prev_r and c_idx_static == empty_prev_c)

                    if not is_moving_tile_original_pos and not is_blank_original_pos:  # only draw non-involved tiles
                        rect_static = get_tile_rect(r_idx_static, c_idx_static)
                        pygame.draw.rect(screen, LIGHT_BLUE, rect_static)
                        pygame.draw.rect(screen, TILE_BORDER_COLOR, rect_static, 3)
                        tile_val_static = board_before_move[r_idx_static][c_idx_static]
                        if tile_val_static != 0:  # Should not be 0 unless error
                            text_surf_static = TILE_FONT.render(str(tile_val_static), True, BLACK)
                            text_rect_static = text_surf_static.get_rect(center=rect_static.center)
                            screen.blit(text_surf_static, text_rect_static)

            # Draw the empty space where the tile is moving from (now looks empty)
            pygame.draw.rect(screen, LIGHT_BLUE, tile_start_rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR, tile_start_rect, 3)

            # Draw the target empty slot (where the tile will land)
            pygame.draw.rect(screen, LIGHT_BLUE, tile_end_rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR, tile_end_rect, 3)

            # Draw the moving tile
            current_tile_x = tile_start_rect.x + dx_tile * frame
            current_tile_y = tile_start_rect.y + dy_tile * frame
            animated_tile_rect = pygame.Rect(current_tile_x, current_tile_y, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, GREEN, animated_tile_rect)  # Moving tile highlighted
            pygame.draw.rect(screen, TILE_BORDER_COLOR, animated_tile_rect, 3)
            text_surf_moving = TILE_FONT.render(str(moving_tile_value), True, BLACK)
            text_rect_moving = text_surf_moving.get_rect(center=animated_tile_rect.center)
            screen.blit(text_surf_moving, text_rect_moving)

            if MSG_FONT:
                anim_msg_surf = MSG_FONT.render(f"Animating step {step_idx + 1}/{len(solution_path) - 1}", True,
                                                GREY)  # Changed message
                anim_msg_rect = anim_msg_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
                screen.blit(anim_msg_surf, anim_msg_rect)
            pygame.display.flip()
            clock.tick(60)
        # current_display_board = copy.deepcopy(next_board_state) # For consistency if used later
    pygame.time.wait(int(max(200, animation_speed * 500)))


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Group 5: 8-Puzzle Solver")
        self.clock = pygame.time.Clock()
        self.running = True

        self.goal_board_str = "1,2,3,4,5,6,7,8,0"
        self.goal_board_state = parse_board_from_string(self.goal_board_str)
        if not self.goal_board_state:
            print("CRITICAL ERROR: Standard default goal is invalid!")
            self.goal_board_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
            self.goal_board_str = "1,2,3,4,5,6,7,8,0"

        self.start_board_str = "1,3,0,4,2,5,7,8,6"
        self.current_board_state = parse_board_from_string(self.start_board_str)

        if not self.current_board_state or \
                not (GiaiDuoc(self.current_board_state) == GiaiDuoc(self.goal_board_state)):
            print(
                f"WARNING: Default start board '{self.start_board_str}' is invalid or unsolvable to goal! Generating random board instead.")
            self.current_board_state = AC3_Generate_Board()
            self.start_board_str = ",".join(str(item) for row in self.current_board_state for item in row)

        self.message = "Select an algorithm. Backtracking has a special rule for tiles 3&6."
        self.buttons = []

        input_width = (BOARD_SIZE - 10) // 2
        self.input_rect_start = pygame.Rect(BOARD_MARGIN_HORIZONTAL, INPUT_BOXES_START_Y, input_width, INPUT_BOX_HEIGHT)
        self.input_rect_goal = pygame.Rect(BOARD_MARGIN_HORIZONTAL + input_width + 10, INPUT_BOXES_START_Y, input_width,
                                           INPUT_BOX_HEIGHT)

        self.setup_buttons()
        self.input_active = None

    def setup_buttons(self):
        button_width = 180
        button_h = BUTTON_HEIGHT
        spacing = 20
        y_row1 = BUTTONS_START_Y
        bt_x1_col1 = (SCREEN_WIDTH - (2 * button_width + spacing)) // 2
        bt_x1_col2 = bt_x1_col1 + button_width + spacing
        self.buttons.append(
            Button(bt_x1_col1, y_row1, button_width, button_h, "Backtracking", BUTTON_COLOR, BUTTON_HOVER_COLOR,
                   BUTTON_TEXT_COLOR, BUTTON_FONT, lambda: self.solve_with_algorithm("Backtracking")))
        self.buttons.append(
            Button(bt_x1_col2, y_row1, button_width, button_h, "Forward Checking", BUTTON_COLOR, BUTTON_HOVER_COLOR,
                   BUTTON_TEXT_COLOR, BUTTON_FONT, lambda: self.solve_with_algorithm("Forward Checking")))
        y_row2 = y_row1 + button_h + BUTTON_ROW_SPACING
        self.buttons.append(
            Button(bt_x1_col1, y_row2, button_width, button_h, "Min-Conflicts", BUTTON_COLOR, BUTTON_HOVER_COLOR,
                   BUTTON_TEXT_COLOR, BUTTON_FONT, lambda: self.solve_with_algorithm("Min-Conflicts")))
        self.buttons.append(
            Button(bt_x1_col2, y_row2, button_width, button_h, "Generate Board AC-3", BUTTON_COLOR, BUTTON_HOVER_COLOR,
                   BUTTON_TEXT_COLOR, BUTTON_FONT, self.generate_new_board))

    def generate_new_board(self):
        self.message = "Generating new board..."  # Changed message
        new_board = AC3_Generate_Board()
        self.current_board_state = new_board
        self.start_board_str = ",".join(str(item) for row in new_board for item in row)
        self.message = "New board generated. Select algorithm or edit board."  # Changed message

    def solve_with_algorithm(self, algo_name):
        parsed_start = parse_board_from_string(self.start_board_str)
        if not parsed_start:
            self.message = "Error: Invalid Start Board!"  # Changed message
            if self.current_board_state:  # Restore string from valid current_board_state if possible
                self.start_board_str = ",".join(str(i) for row in self.current_board_state for i in row)
            return

        self.current_board_state = parsed_start  # Update current board with the parsed one
        initial_message = f"Solving with {algo_name}..."  # Changed message
        if algo_name == "Backtracking":
            initial_message = f"Solving with Backtracking (prioritizing keeping 3&6 adjacent if already so)..."  # Changed message

        if not GiaiDuoc(self.current_board_state) == GiaiDuoc(self.goal_board_state):
            warning_prefix = "Warning: Board may not be solvable to standard goal!"  # Changed message
            if algo_name == "Backtracking":
                initial_message = f"{warning_prefix} Trying with Backtracking (prioritizing keeping 3&6 adjacent)..."  # Changed message
            else:
                initial_message = f"{warning_prefix} Trying with {algo_name}..."  # Changed message

        self.message = initial_message
        self.draw()  # Show "Solving..." message
        pygame.display.flip()

        solution_path = None
        start_time = time.time()
        current_limit_val = 0  # To store the limit value used for messaging

        if algo_name == "Backtracking":
            current_limit_val = BACKTRACKING_MAX_DEPTH
            solution_path = Backtracking(self.current_board_state, self.goal_board_state, depth=current_limit_val)
        elif algo_name == "Forward Checking":
            current_limit_val = FORWARD_CHECKING_MAX_DEPTH
            solution_path = Forward_Checking(self.current_board_state, self.goal_board_state, depth=current_limit_val)
        elif algo_name == "Min-Conflicts":
            current_limit_val = MIN_CONFLICTS_MAX_ITERATIONS
            solution_path = Min_Conflicts_Search(self.current_board_state, self.goal_board_state,
                                                 max_iterations=current_limit_val)

        solve_time = time.time() - start_time

        if solution_path and solution_path[-1] == self.goal_board_state:
            self.current_board_state = solution_path[-1]  # Update board to goal state
            self.start_board_str = ",".join(
                str(i) for r in self.current_board_state for i in r)  # Reflect final state in input box
            num_moves = len(solution_path) - 1
            self.message = f"{algo_name}: Solved! {num_moves} steps ({solve_time:.2f}s). Animating..."  # Changed message
            self.draw()  # Update message before animation
            pygame.display.flip()
            animate_solution_path(self.screen, self.clock, solution_path[0], solution_path)
            self.message = f"{algo_name}: Solved! {num_moves} steps in {solve_time:.2f}s. Board is at goal state."  # Changed message
        elif solution_path:  # Path found but not to goal (likely hit iteration/depth limit)
            self.current_board_state = solution_path[-1]  # Update board to the last state reached
            self.start_board_str = ",".join(str(i) for r in self.current_board_state for i in r)  # Reflect final state
            full_path_moves = len(solution_path) - 1 if len(solution_path) > 0 else 0

            limit_reason_detail = f"after {current_limit_val} iterations" if algo_name == "Min-Conflicts" else "limit reached"

            path_to_animate = solution_path
            animated_moves_count = full_path_moves
            temp_animation_message = f"{algo_name}: Goal not reached ({limit_reason_detail}). Path: {full_path_moves} steps ({solve_time:.2f}s)."  # Changed message

            if full_path_moves > MAX_ANIMATION_STEPS_NO_GOAL:
                path_to_animate = solution_path[:MAX_ANIMATION_STEPS_NO_GOAL + 1]
                animated_moves_count = len(path_to_animate) - 1 if len(path_to_animate) > 0 else 0
                temp_animation_message = (
                    f"{algo_name}: Goal not reached ({limit_reason_detail}). Animating {animated_moves_count}/{full_path_moves} steps.")  # Changed message

            self.message = temp_animation_message + " Animating..."  # Changed message
            self.draw()
            pygame.display.flip()
            if path_to_animate and len(path_to_animate) > 1:
                animate_solution_path(self.screen, self.clock, path_to_animate[0], path_to_animate)
            self.message = f"{algo_name}: Goal not reached ({limit_reason_detail}). Path: {full_path_moves} steps in {solve_time:.2f}s. Board at final state."  # Changed message
        else:  # No solution path found at all
            # Use current_limit_val to make message more specific
            algo_limit_map = {
                "Backtracking": f"due to depth limit {current_limit_val} steps",
                "Forward Checking": f"due to depth limit {current_limit_val} steps"
            }
            limit_reason_detail = algo_limit_map.get(algo_name, "(limit possibly reached)")
            self.message = f"{algo_name}: No solution found ({limit_reason_detail}). Time: {solve_time:.2f}s. Board at initial state."  # Changed message

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # When this window is closed, launch main.py
                self.running = False
                pygame.quit()
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
                os.system(f'python "{script_path}"')
                sys.exit()
    
            for button in self.buttons:
                button.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.input_rect_start.collidepoint(event.pos):
                    self.input_active = 'start'
                    self.message = "Edit Start board (e.g., 1,2,3,4,0,5,6,7,8)"
                elif self.input_rect_goal.collidepoint(event.pos):
                    self.input_active = 'goal' 
                    self.message = "Edit Goal board (e.g., 1,2,3,4,5,6,7,8,0)"
                elif not any(b.rect.collidepoint(event.pos) for b in self.buttons):
                    self.input_active = None

            if event.type == pygame.KEYDOWN and self.input_active:
                active_str_ref = 'start_board_str' if self.input_active == 'start' else 'goal_board_str'
                current_text = getattr(self, active_str_ref)

                if event.key == pygame.K_RETURN:
                    parsed_board = parse_board_from_string(current_text)
                    valid_input = False
                    user_error_message = ""

                    if not parsed_board:
                        user_error_message = f"Error: Invalid {'Start' if self.input_active == 'start' else 'Goal'} board format!"  # Changed message
                    else:
                        valid_input = True
                        if self.input_active == 'start':
                            self.current_board_state = parsed_board
                            setattr(self, active_str_ref, current_text)  # Keep user's valid text
                            self.message = "Start board updated."  # Changed message
                            # Check solvability against current goal
                            if not GiaiDuoc(parsed_board) == GiaiDuoc(self.goal_board_state):
                                self.message += " Warning: May not be solvable to current goal."  # Changed message
                        else:  # 'goal'
                            self.goal_board_state = parsed_board
                            setattr(self, active_str_ref, current_text)  # Keep user's valid text
                            self.message = "Goal board updated."  # Changed message
                            # Check solvability of current start against new goal
                            if not GiaiDuoc(self.current_board_state) == GiaiDuoc(parsed_board):
                                self.message += " Warning: Current start may not be solvable to this new goal."

                    if not valid_input:  # If input was bad, revert the text field
                        original_board_obj = self.current_board_state if self.input_active == 'start' else self.goal_board_state
                        setattr(self, active_str_ref, ",".join(str(i) for r in original_board_obj for i in r))
                        self.message = user_error_message + " Reverted."  # Changed message
                    self.input_active = None  # Deactivate on enter

                elif event.key == pygame.K_BACKSPACE:
                    setattr(self, active_str_ref, current_text[:-1])
                elif event.unicode.isdigit() or event.unicode == ',':
                    if len(current_text) < 17:  # Max length for "1,2,3,4,5,6,7,8,0"
                        setattr(self, active_str_ref, current_text + event.unicode)

    def draw_input_boxes(self):
        start_label_text = "Start (e.g., 1,2,3,4,0,5,6,7,8):"  # Changed message
        if BUTTON_FONT:
            start_label_surf = BUTTON_FONT.render(start_label_text, True, BLACK)
            self.screen.blit(start_label_surf,
                             (self.input_rect_start.x, self.input_rect_start.y - INPUT_LABEL_OFFSET_Y))
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.input_rect_start)
        start_border_color = GREEN if self.input_active == 'start' else GREY
        pygame.draw.rect(self.screen, start_border_color, self.input_rect_start, 2, border_radius=3)
        if BUTTON_FONT:
            start_text_surf = BUTTON_FONT.render(self.start_board_str, True, BLACK)
            self.screen.blit(start_text_surf, (self.input_rect_start.x + 5, self.input_rect_start.y + (
                    self.input_rect_start.height - BUTTON_FONT.get_height()) // 2))

        goal_label_text = "Goal (e.g., 1,2,3,4,5,6,7,8,0):"  # Changed message
        if BUTTON_FONT:
            goal_label_surf = BUTTON_FONT.render(goal_label_text, True, BLACK)
            self.screen.blit(goal_label_surf, (self.input_rect_goal.x, self.input_rect_goal.y - INPUT_LABEL_OFFSET_Y))
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.input_rect_goal)
        goal_border_color = GREEN if self.input_active == 'goal' else GREY
        pygame.draw.rect(self.screen, goal_border_color, self.input_rect_goal, 2, border_radius=3)
        if BUTTON_FONT:
            goal_text_surf = BUTTON_FONT.render(self.goal_board_str, True, BLACK)
            self.screen.blit(goal_text_surf, (self.input_rect_goal.x + 5, self.input_rect_goal.y + (
                    self.input_rect_goal.height - BUTTON_FONT.get_height()) // 2))

    def draw(self):
        self.screen.fill(WHITE)
        if not (self.current_board_state and DoiMotKhacNhau(self.current_board_state)):
            # This might happen if start_board_str was corrupted somehow without parsing check
            self.current_board_state = AC3_Generate_Board()  # Regenerate a valid board
            self.start_board_str = ",".join(str(i) for r in self.current_board_state for i in r)
            self.message = "Error: Invalid board state. Reset."  # Changed message

        draw_board_static(self.screen, self.current_board_state)
        draw_main_message(self.screen, self.message, MESSAGE_Y_CENTER)
        for button in self.buttons:
            button.draw(self.screen)
        self.draw_input_boxes()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()  # Draw happens every frame
            self.clock.tick(30)
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game = Game()
    game.run()
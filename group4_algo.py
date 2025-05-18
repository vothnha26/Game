import pygame
import random
import copy
import heapq

# --- Common Constants ---
GRID_SIZE = 3
DISPLAY_TILE_SIZE_PO = 25
DISPLAY_TILE_SIZE_NON_OBS = 20
DISPLAY_TILE_SIZE_AND_OR = 60

# --- New Color Palette ---
APP_BACKGROUND_COLOR = (240, 242, 245)  # Light grayish blue
CONTENT_BACKGROUND_COLOR = (220, 223, 228)  # Slightly darker gray for content
HEADER_BG_COLOR = (40, 50, 70)  # Dark slate blue
TEXT_COLOR_DARK = (30, 35, 40)  # Dark gray for text on light backgrounds
TEXT_COLOR_LIGHT = (240, 240, 240)  # Light gray/white for text on dark backgrounds

TILE_COLOR = (70, 130, 180)  # Steel Blue
TILE_EMPTY_COLOR = (190, 195, 200)  # Light gray for empty tile
TILE_BORDER_COLOR_NEW = (50, 90, 130)  # Darker blue border
TILE_BORDER_WIDTH_NEW = 1
TILE_BORDER_RADIUS = 3

BUTTON_BG_COLOR = (80, 120, 170)  # A nice blue
BUTTON_HOVER_COLOR = (100, 140, 190)  # Lighter blue for hover
BUTTON_ACTIVE_COLOR = (60, 100, 150)  # Slightly darker for active/pressed
BUTTON_TEXT_COLOR_NEW = TEXT_COLOR_LIGHT
DISABLED_BUTTON_BG_COLOR = (160, 170, 180)  # Muted gray for disabled

# Highlights
HIGHLIGHT_GREEN_CORRECT_POS = (60, 179, 113, 200)  # Medium Sea Green (alpha for surface)
HIGHLIGHT_FIXED_TILE = (100, 150, 235, 150)  # A lighter blue for fixed tiles (alpha)
HIGHLIGHT_YELLOW_PO_ROW = (255, 223, 100, 128)  # Softer Yellow (alpha)
HIGHLIGHT_SOLVED_PUZZLE = (70, 180, 120, 180)  # Calm Green (alpha)
HIGHLIGHT_FOUND_GOAL_BORDER = (255, 190, 0)  # Gold/Amber for found goal border
HIGHLIGHT_AND_OR_FIXED_BORDER = (255, 100, 0)  # Orange for AND-OR fixed border

# --- Main Screen Configuration ---
SCREEN_WIDTH_MAIN = 800
SCREEN_HEIGHT_MAIN = 700
HEADER_HEIGHT = 50
CONTENT_Y_OFFSET = HEADER_HEIGHT + 10


# --- Application States ---
class GameState:
    PARTIALLY_OBSERVABLE = 1
    AND_OR_SEARCH_DEMO = 2
    NON_OBSERVABLE_AUTO_RANDOM = 3


# --- Goal States ---
STANDARD_GOAL_STATE_LIST = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
STANDARD_GOAL_STATE_TUPLE = tuple(map(tuple, STANDARD_GOAL_STATE_LIST))
PARTIAL_OBSERVABLE_GOAL_FIXED_PART = [[1, 2, 3]]

TARGET_POSITIONS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)
}


# --- Description Functions ---
def get_Nhom4_TreeSearch_AND_OR_description():
    return [
        "AND-OR Search Demo (Next 'Easiest' Subgoal):", "",
        "Goal: Move tiles to their correct positions one by one.",
        "After a tile is correctly placed, it becomes 'fixed'.",
        "The system will select the 'easiest' unfixed tile (closest to its goal) as the next target.",
        "Click 'Solve Easiest Subgoal'."
    ]


def get_Nhom4_PartiallyObservable_description():
    return [
        "Partially Observable (Automatic Random Actions):", "",
        "The top row [1,2,3] is always fixed.",
        "The system automatically performs random actions for all belief matrices.",
        "The process stops when one of the matrices reaches the goal state.",
        "Press SPACE to Pause/Resume, 'R' to Reset."
    ]


def get_Nhom4_NonObservable_Auto_Random_description():
    return [
        "Non-Observable (Automatic Random Actions):", "",
        "Displays a large set of standard 8-puzzle problems.",
        "These problems are initialized randomly but close to the goal state.",
        "The program automatically performs random actions,",
        "applied simultaneously to all problems.",
        "The process stops when one of the problems reaches the goal state.",
        "Tiles in their correct goal positions will be highlighted green."
    ]


# --- General Helpers ---
def find_zero_pos(state_list_or_tuple):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state_list_or_tuple[r][c] == 0: return (r, c)
    return None


def get_tile_pos(state_list_or_tuple, tile_value):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state_list_or_tuple[r][c] == tile_value: return (r, c)
    return None


def manhattan_distance(pos1, pos2):
    if pos1 is None or pos2 is None: return float('inf')
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# --- Logic for Standard Puzzle (Used for Non-Obs and AND-OR) ---
def get_valid_successors_standard(state_tuple):
    successors = []
    zero_pos = find_zero_pos(state_tuple)
    if not zero_pos: return successors
    zero_r, zero_c = zero_pos
    possible_moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    current_state_list = [list(row) for row in state_tuple]
    for dr, dc, action_name in possible_moves:
        next_zero_r, next_zero_c = zero_r + dr, zero_c + dc
        if 0 <= next_zero_r < GRID_SIZE and 0 <= next_zero_c < GRID_SIZE:
            new_state_list = copy.deepcopy(current_state_list)
            new_state_list[zero_r][zero_c], new_state_list[next_zero_r][next_zero_c] = new_state_list[next_zero_r][
                next_zero_c], new_state_list[zero_r][zero_c]
            successors.append({'action': action_name, 'state': tuple(map(tuple, new_state_list))})
    return successors


def apply_action_standard(state_list, action_name):
    zero_pos = find_zero_pos(state_list)
    if not zero_pos: return copy.deepcopy(state_list)
    zero_r, zero_c = zero_pos
    dr, dc = 0, 0
    if action_name == 'UP':
        dr = -1
    elif action_name == 'DOWN':
        dr = 1
    elif action_name == 'LEFT':
        dc = -1
    elif action_name == 'RIGHT':
        dc = 1
    else:
        return copy.deepcopy(state_list)
    next_zero_r, next_zero_c = zero_r + dr, zero_c + dc
    new_state = copy.deepcopy(state_list)
    if 0 <= next_zero_r < GRID_SIZE and 0 <= next_zero_c < GRID_SIZE:
        new_state[zero_r][zero_c], new_state[next_zero_r][next_zero_c] = new_state[next_zero_r][next_zero_c], \
            new_state[zero_r][zero_c]
    return new_state


def generate_initial_states_standard_close(num_puzzles=1, root_shuffles=5, variation_shuffles=0):
    puzzles = []
    generated_tuples = set()
    for _ in range(num_puzzles * 3):  # Try more times to get unique puzzles
        if len(puzzles) >= num_puzzles: break
        current_state_tuple = STANDARD_GOAL_STATE_TUPLE
        total_shuffles = root_shuffles + random.randint(0, variation_shuffles)
        for _ in range(total_shuffles):
            successors = get_valid_successors_standard(current_state_tuple)
            if not successors: break
            current_state_tuple = random.choice(successors)['state']
        if current_state_tuple not in generated_tuples:
            puzzles.append([list(row) for row in current_state_tuple])
            generated_tuples.add(current_state_tuple)
    # Fallback for not enough unique puzzles
    while len(puzzles) < num_puzzles:
        if puzzles:  # Try to derive from existing if possible
            base_for_fallback = tuple(map(tuple, puzzles[random.randint(0, len(puzzles) - 1)]))
            for _ in range(max(1, variation_shuffles if variation_shuffles > 0 else 1)):  # Shuffle a bit more
                successors = get_valid_successors_standard(base_for_fallback)
                if not successors: break
                base_for_fallback = random.choice(successors)['state']
            if base_for_fallback not in generated_tuples:
                puzzles.append([list(row) for row in base_for_fallback])
                generated_tuples.add(base_for_fallback)
            else:  # Still duplicate, just copy first one
                puzzles.append(copy.deepcopy(puzzles[0]))  # Or generate a completely new one
        else:  # No puzzles at all, start from goal
            puzzles.append(copy.deepcopy(STANDARD_GOAL_STATE_LIST))
        if len(puzzles) >= num_puzzles: break
    return puzzles[:num_puzzles]


# --- Logic for Partially Observable (PO) ---
def get_valid_successors_for_bfs_po(state_tuple):
    successors = []
    zero_r, zero_c = find_zero_pos(state_tuple)
    possible_moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    current_state_list = [list(row) for row in state_tuple]
    for dr, dc, action in possible_moves:
        next_zero_r, next_zero_c = zero_r + dr, zero_c + dc
        if 0 <= next_zero_r < GRID_SIZE and 0 <= next_zero_c < GRID_SIZE and next_zero_r != 0:  # PO constraint
            new_state_list = copy.deepcopy(current_state_list)
            new_state_list[zero_r][zero_c], new_state_list[next_zero_r][next_zero_c] = \
                new_state_list[next_zero_r][next_zero_c], new_state_list[zero_r][zero_c]
            successors.append({'action': action, 'state': tuple(map(tuple, new_state_list))})
    return successors


def apply_single_action_to_state_po(state_list, action):
    zero_pos = find_zero_pos(state_list)
    if not zero_pos: return copy.deepcopy(state_list)
    zero_r, zero_c = zero_pos
    dr, dc = 0, 0
    if action == 'UP':
        dr, dc = -1, 0
    elif action == 'DOWN':
        dr, dc = 1, 0
    elif action == 'LEFT':
        dr, dc = 0, -1
    elif action == 'RIGHT':
        dr, dc = 0, 1
    else:
        return copy.deepcopy(state_list)

    next_zero_r, next_zero_c = zero_r + dr, zero_c + dc
    new_state = copy.deepcopy(state_list)
    if 0 <= next_zero_r < GRID_SIZE and 0 <= next_zero_c < GRID_SIZE and next_zero_r != 0:  # PO constraint
        new_state[zero_r][zero_c], new_state[next_zero_r][next_zero_c] = \
            new_state[next_zero_r][next_zero_c], new_state[zero_r][zero_c]
    return new_state


def generate_solvable_close_states_po(num_states=9, max_shuffles=6, min_shuffles=2):
    belief_set = []
    generated_states_tuples = set()
    for _ in range(num_states * 3):  # Attempt more to get unique states
        if len(belief_set) >= num_states: break
        current_state_tuple = STANDARD_GOAL_STATE_TUPLE
        num_shuffles = random.randint(min_shuffles, max_shuffles)
        for _ in range(num_shuffles):
            successors = get_valid_successors_for_bfs_po(current_state_tuple)  # Use PO-valid moves for shuffling
            if not successors: break
            current_state_tuple = random.choice(successors)['state']

        current_state_list = [list(row) for row in current_state_tuple]
        if current_state_list[0] != PARTIAL_OBSERVABLE_GOAL_FIXED_PART[0]:
            continue  # Should not happen if get_valid_successors_for_bfs_po is correct

        if current_state_tuple not in generated_states_tuples and current_state_tuple != STANDARD_GOAL_STATE_TUPLE:
            belief_set.append(current_state_list)
            generated_states_tuples.add(current_state_tuple)

    while len(belief_set) < num_states:  # Fallback if not enough unique states generated
        fixed_top_row = PARTIAL_OBSERVABLE_GOAL_FIXED_PART[0][:]
        remaining_numbers = [0, 4, 5, 6, 7, 8]
        random.shuffle(remaining_numbers)  # Shuffle remaining numbers
        state_m = [fixed_top_row,
                   remaining_numbers[0:GRID_SIZE],
                   remaining_numbers[GRID_SIZE:GRID_SIZE * 2]]
        state_t = tuple(map(tuple, state_m))
        if state_t not in generated_states_tuples:
            belief_set.append(state_m)
            generated_states_tuples.add(state_t)
        elif belief_set:  # If still struggling, duplicate an existing one
            belief_set.append(copy.deepcopy(belief_set[0]))
        else:  # Absolute fallback
            belief_set.append(copy.deepcopy(STANDARD_GOAL_STATE_LIST))
            break  # Should not happen
    return belief_set[:num_states]


def draw_belief_state_po(screen, font, state, x_offset, y_offset, tile_visual_size, is_fully_solved,
                         is_partially_consistent, is_found_goal_overall=False):
    base_rect_for_highlight = pygame.Rect(x_offset, y_offset, tile_visual_size * GRID_SIZE,
                                          tile_visual_size * GRID_SIZE)
    if is_found_goal_overall and is_fully_solved:
        pygame.draw.rect(screen, HIGHLIGHT_FOUND_GOAL_BORDER, base_rect_for_highlight.inflate(6, 6), 3,
                         border_radius=TILE_BORDER_RADIUS + 2)

    for r_idx, row in enumerate(state):
        for c_idx, num in enumerate(row):
            rect = pygame.Rect(x_offset + c_idx * tile_visual_size, y_offset + r_idx * tile_visual_size,
                               tile_visual_size, tile_visual_size)
            bg_color = TILE_EMPTY_COLOR if num == 0 else TILE_COLOR
            pygame.draw.rect(screen, bg_color, rect, border_radius=TILE_BORDER_RADIUS)

            highlight_surface = pygame.Surface((tile_visual_size, tile_visual_size), pygame.SRCALPHA)
            # highlight_drawn = False # Variable not used
            if is_fully_solved:
                highlight_surface.fill(HIGHLIGHT_SOLVED_PUZZLE)
                screen.blit(highlight_surface, (rect.x, rect.y))
                # highlight_drawn = True
            elif is_partially_consistent and r_idx == 0 and state[0] == PARTIAL_OBSERVABLE_GOAL_FIXED_PART[0]:
                highlight_surface.fill(HIGHLIGHT_YELLOW_PO_ROW)
                screen.blit(highlight_surface, (rect.x, rect.y))
                # highlight_drawn = True

            if num != 0:
                text_surf = font.render(str(num), True, TEXT_COLOR_LIGHT)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR_NEW, rect, TILE_BORDER_WIDTH_NEW,
                             border_radius=TILE_BORDER_RADIUS)


def check_goal_state_standard(state_list): return state_list == STANDARD_GOAL_STATE_LIST


def check_consistent_with_goal_fixed_part_po(state_list): return state_list[0] == PARTIAL_OBSERVABLE_GOAL_FIXED_PART[0]


class PartiallyObservableSim:
    def __init__(self, content_rect):
        self.content_rect = content_rect
        self.tile_font = pygame.font.Font(None, int(DISPLAY_TILE_SIZE_PO * 0.7))
        self.info_font = pygame.font.Font(None, 22)
        self.status_font = pygame.font.Font(None, 24)

        self.initial_num_belief_states = 9
        self.belief_states_cols_display = 3
        self.belief_states_rows_display = (
                                                      self.initial_num_belief_states + self.belief_states_cols_display - 1) // self.belief_states_cols_display
        self.belief_visual_width = GRID_SIZE * DISPLAY_TILE_SIZE_PO
        self.belief_visual_height = GRID_SIZE * DISPLAY_TILE_SIZE_PO
        self.padding_belief = 10
        self.margin_sim_area = 15
        self.sim_top_info_height = 40

        self.AUTO_STEP_DELAY_PO = 300
        self.reset_simulation_full()

    def reset_simulation_full(self):
        self.current_belief_set = generate_solvable_close_states_po(self.initial_num_belief_states, max_shuffles=8,
                                                                    min_shuffles=3)
        self.step_count = 0
        self.last_action_performed = "None"
        self.auto_random_active = True
        self.auto_step_timer = pygame.time.get_ticks()
        self.goal_found_indices_po = []
        self.status_message = "Starting random actions... ('R': Reset, SPACE: Pause/Resume)"

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_simulation_full()
                return True
            if event.key == pygame.K_SPACE:
                self.auto_random_active = not self.auto_random_active
                if self.auto_random_active and not self.goal_found_indices_po:
                    self.status_message = f"Resuming... (Step {self.step_count})"
                    self.auto_step_timer = pygame.time.get_ticks()
                elif not self.auto_random_active:
                    self.status_message = f"Paused at step {self.step_count}. SPACE to resume."
                elif self.goal_found_indices_po:
                    self.status_message = f"Goal found! Press 'R' to Play Again."
                return True
        return False

    def perform_random_step_po(self):
        if not self.auto_random_active or self.goal_found_indices_po:
            return

        self.step_count += 1
        action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.last_action_performed = action

        new_belief_set_temp = []

        for i, state_list in enumerate(self.current_belief_set):
            if i in self.goal_found_indices_po:
                new_belief_set_temp.append(copy.deepcopy(state_list))
                continue

            new_s = apply_single_action_to_state_po(state_list, action)
            new_belief_set_temp.append(new_s)

        self.current_belief_set = new_belief_set_temp
        self.check_for_goal_and_stop_po()

    def check_for_goal_and_stop_po(self):
        if not self.auto_random_active and self.goal_found_indices_po: # Check if already paused due to goal
            return

        any_new_goal_this_step = False
        for i, p_state in enumerate(self.current_belief_set):
            if i not in self.goal_found_indices_po:
                if check_goal_state_standard(p_state):
                    self.goal_found_indices_po.append(i)
                    any_new_goal_this_step = True

        if any_new_goal_this_step:
            self.auto_random_active = False # Stop auto stepping
            found_indices_str = ", ".join(str(idx + 1) for idx in self.goal_found_indices_po)
            self.status_message = f"GOAL in matrix {found_indices_str} after {self.step_count} steps! (Last: {self.last_action_performed})"
        elif self.auto_random_active: # Only update if still active (no goal found this step)
             self.status_message = f"Step {self.step_count} after '{self.last_action_performed}'. SPACE: Pause/Resume."


    def update(self):
        if self.auto_random_active and not self.goal_found_indices_po:
            current_time = pygame.time.get_ticks()
            if current_time - self.auto_step_timer >= self.AUTO_STEP_DELAY_PO:
                self.perform_random_step_po()
                self.auto_step_timer = current_time

    def draw(self, screen, mouse_pos):
        pygame.draw.rect(screen, CONTENT_BACKGROUND_COLOR, self.content_rect)

        status_s = self.status_font.render(self.status_message, True, TEXT_COLOR_DARK)
        status_r = status_s.get_rect(centerx=self.content_rect.centerx, top=self.content_rect.top + 10)
        screen.blit(status_s, status_r)

        draw_area_start_y = self.content_rect.top + self.sim_top_info_height + 5
        for i in range(len(self.current_belief_set)):
            if i >= self.initial_num_belief_states: break

            row_idx = i // self.belief_states_cols_display
            col_idx = i % self.belief_states_cols_display
            start_x = self.content_rect.left + self.margin_sim_area + col_idx * (
                        self.belief_visual_width + self.padding_belief)
            start_y = draw_area_start_y + row_idx * (self.belief_visual_height + self.padding_belief)

            state_to_draw = self.current_belief_set[i]
            is_fully_solved_flag = check_goal_state_standard(state_to_draw)
            is_part_consistent = check_consistent_with_goal_fixed_part_po(state_to_draw)
            is_this_one_found_goal = (i in self.goal_found_indices_po)

            draw_belief_state_po(screen, self.tile_font, state_to_draw, start_x, start_y,
                                 DISPLAY_TILE_SIZE_PO, is_fully_solved_flag, is_part_consistent, is_this_one_found_goal)


# --- Logic for Non-Observable Auto Random (NOAR) ---
def draw_single_puzzle_highlight_correct(screen, font, puzzle_state, goal_state_list, x, y, tile_size,
                                         is_fully_solved_board=False):
    base_rect_for_board_highlight = pygame.Rect(x, y, tile_size * GRID_SIZE, tile_size * GRID_SIZE)
    if is_fully_solved_board:
        pygame.draw.rect(screen, HIGHLIGHT_FOUND_GOAL_BORDER, base_rect_for_board_highlight.inflate(6, 6), 3,
                         border_radius=TILE_BORDER_RADIUS + 2)

    for r_idx, row in enumerate(puzzle_state):
        for c_idx, num in enumerate(row):
            rect = pygame.Rect(x + c_idx * tile_size, y + r_idx * tile_size, tile_size, tile_size)
            is_correct_pos_tile = (num != 0 and num == goal_state_list[r_idx][c_idx])

            current_bg_color = TILE_EMPTY_COLOR if num == 0 else TILE_COLOR
            pygame.draw.rect(screen, current_bg_color, rect, border_radius=TILE_BORDER_RADIUS)

            highlight_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
            # highlighted_this_tile = False # Variable not used

            if is_fully_solved_board:
                if num != 0:
                    highlight_surface.fill(HIGHLIGHT_SOLVED_PUZZLE)
                    screen.blit(highlight_surface, rect.topleft)
                    # highlighted_this_tile = True
            elif is_correct_pos_tile:
                highlight_surface.fill(HIGHLIGHT_GREEN_CORRECT_POS)
                screen.blit(highlight_surface, rect.topleft)
                # highlighted_this_tile = True

            if num != 0:
                text_surf = font.render(str(num), True, TEXT_COLOR_LIGHT)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR_NEW, rect, TILE_BORDER_WIDTH_NEW,
                             border_radius=TILE_BORDER_RADIUS)


class NonObservableAutoRandomSim:
    def __init__(self, content_rect):
        self.content_rect = content_rect
        self.tile_font = pygame.font.Font(None, int(DISPLAY_TILE_SIZE_NON_OBS * 0.7))
        self.status_font = pygame.font.Font(None, 24)
        self.num_puzzles = 32
        self.cols = 8
        self.rows = (self.num_puzzles + self.cols - 1) // self.cols
        self.puzzle_visual_width = GRID_SIZE * DISPLAY_TILE_SIZE_NON_OBS
        self.puzzle_visual_height = GRID_SIZE * DISPLAY_TILE_SIZE_NON_OBS
        self.padding = 5
        self.margin_sim_area = 10
        self.sim_top_info_height = 35
        self.AUTO_STEP_DELAY = 200
        self.reset_simulation()

    def reset_simulation(self):
        self.puzzles = generate_initial_states_standard_close(self.num_puzzles, root_shuffles=20, variation_shuffles=5)
        self.step_count = 0
        self.last_action_performed = "None"
        self.auto_solving_active = True
        self.auto_step_timer = pygame.time.get_ticks()
        self.goal_found_puzzle_indices = []
        self.status_message = "Starting automatic scenario... ('R': Reset, SPACE: Pause/Resume)"

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_simulation()
                return True
            if event.key == pygame.K_SPACE:
                self.auto_solving_active = not self.auto_solving_active
                if self.auto_solving_active and not self.goal_found_puzzle_indices:
                    self.status_message = f"Resuming... (Step {self.step_count})"
                    self.auto_step_timer = pygame.time.get_ticks()
                elif not self.auto_solving_active:
                    self.status_message = f"Paused at step {self.step_count}. SPACE to resume."
                elif self.goal_found_puzzle_indices:
                    self.status_message = f"Goal found! Press 'R' to Play Again."
                return True
        return False

    def perform_random_step(self):
        if not self.auto_solving_active or self.goal_found_puzzle_indices:
            return

        self.step_count += 1
        action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.last_action_performed = action

        any_new_goal_found_this_step = False
        for i in range(len(self.puzzles)):
            if i in self.goal_found_puzzle_indices:
                continue
            self.puzzles[i] = apply_action_standard(self.puzzles[i], action)
            if check_goal_state_standard(self.puzzles[i]):
                if i not in self.goal_found_puzzle_indices:
                    self.goal_found_puzzle_indices.append(i)
                    any_new_goal_found_this_step = True

        if any_new_goal_found_this_step:
            self.auto_solving_active = False # Stop auto stepping
            found_indices_str = ", ".join(str(idx + 1) for idx in self.goal_found_puzzle_indices)
            self.status_message = f"GOAL in matrix {found_indices_str} after {self.step_count} steps! (Last: {self.last_action_performed})"
        elif self.auto_solving_active: # Only update if still active
            self.status_message = f"Step {self.step_count} after '{self.last_action_performed}'. SPACE: Pause/Resume."


    def update(self):
        if self.auto_solving_active and not self.goal_found_puzzle_indices:
            current_time = pygame.time.get_ticks()
            if current_time - self.auto_step_timer >= self.AUTO_STEP_DELAY:
                self.perform_random_step()
                self.auto_step_timer = current_time

    def draw(self, screen, mouse_pos):
        pygame.draw.rect(screen, CONTENT_BACKGROUND_COLOR, self.content_rect)
        status_surf = self.status_font.render(self.status_message, True, TEXT_COLOR_DARK)
        status_rect = status_surf.get_rect(centerx=self.content_rect.centerx, top=self.content_rect.top + 10)
        screen.blit(status_surf, status_rect)

        draw_area_start_y = self.content_rect.top + self.sim_top_info_height + 5
        total_grid_width = self.cols * self.puzzle_visual_width + max(0, self.cols - 1) * self.padding
        offset_x_grid = (self.content_rect.width - total_grid_width) // 2

        for i, p_state in enumerate(self.puzzles):
            row_idx = i // self.cols
            col_idx = i % self.cols
            start_x = self.content_rect.left + offset_x_grid + col_idx * (self.puzzle_visual_width + self.padding)
            start_y = draw_area_start_y + row_idx * (self.puzzle_visual_height + self.padding)
            if start_y + self.puzzle_visual_height > self.content_rect.bottom - self.padding:
                continue
            is_found_at_goal = (i in self.goal_found_puzzle_indices)
            draw_single_puzzle_highlight_correct(screen, self.tile_font, p_state, STANDARD_GOAL_STATE_LIST,
                                                 start_x, start_y, DISPLAY_TILE_SIZE_NON_OBS, is_found_at_goal)


# --- Logic for AND-OR Search Demo ---
class AndOrSearchSim:
    def __init__(self, content_rect):
        self.content_rect = content_rect
        self.tile_font = pygame.font.Font(None, int(DISPLAY_TILE_SIZE_AND_OR * 0.6))
        self.status_font = pygame.font.Font(None, 28)
        self.button_font = pygame.font.Font(None, 26)
        self.puzzle_visual_width = GRID_SIZE * DISPLAY_TILE_SIZE_AND_OR
        self.puzzle_visual_height = GRID_SIZE * DISPLAY_TILE_SIZE_AND_OR
        self.margin_sim_area = 20
        self.sim_top_info_height = 80
        self.solve_button_rect = pygame.Rect(self.content_rect.centerx - 130, self.content_rect.top + 15, 260, 45)
        self.solve_button_text_default = "Solve 'Easiest' Subgoal"
        self.animation_delay_per_step = 200
        self.reset_simulation()

    def reset_simulation(self):
        self.puzzle_state = generate_initial_states_standard_close(1, root_shuffles=random.randint(3, 6))[0]
        self.fixed_tiles = {}
        self.animation_path = []
        self.current_anim_step = 0
        self.is_animating = False
        self.all_subgoals_completed = False
        self.current_target_tile_and_pos = None
        self.determine_next_easiest_subgoal()

    def determine_next_easiest_subgoal(self):
        self.current_target_tile_and_pos = None
        min_dist_heuristic = float('inf')
        easiest_tile_to_place = None

        for tile_num_check in range(1, GRID_SIZE * GRID_SIZE):
            target_pos_check = TARGET_POSITIONS.get(tile_num_check)
            if not target_pos_check: continue
            if tile_num_check not in self.fixed_tiles:
                if get_tile_pos(self.puzzle_state, tile_num_check) == target_pos_check:
                    self.fixed_tiles[tile_num_check] = target_pos_check

        for tile_num_to_check in range(1, GRID_SIZE * GRID_SIZE):
            target_pos_for_tile = TARGET_POSITIONS.get(tile_num_to_check)
            if not target_pos_for_tile: continue
            if tile_num_to_check in self.fixed_tiles and self.fixed_tiles[tile_num_to_check] == target_pos_for_tile:
                continue

            current_pos_of_tile = get_tile_pos(self.puzzle_state, tile_num_to_check)
            if not current_pos_of_tile: continue
            if current_pos_of_tile == target_pos_for_tile:
                self.fixed_tiles[tile_num_to_check] = target_pos_for_tile
                continue

            dist_tile_to_target = manhattan_distance(current_pos_of_tile, target_pos_for_tile)
            zero_pos = find_zero_pos(self.puzzle_state)
            dist_blank_to_tile = manhattan_distance(zero_pos, current_pos_of_tile) if zero_pos else float('inf')
            current_heuristic = dist_tile_to_target + (0 if dist_blank_to_tile == 1 else 1) + (dist_blank_to_tile * 0.1)

            if current_heuristic < min_dist_heuristic:
                min_dist_heuristic = current_heuristic
                easiest_tile_to_place = tile_num_to_check

        if easiest_tile_to_place is not None:
            self.current_target_tile_and_pos = (easiest_tile_to_place, TARGET_POSITIONS[easiest_tile_to_place])
            tile, (r, c) = self.current_target_tile_and_pos
            self.status_message = f"Subgoal: Place tile {tile} at ({r},{c})"
        else:
            blank_target_pos = TARGET_POSITIONS.get(0)
            current_blank_pos = find_zero_pos(self.puzzle_state)
            if current_blank_pos == blank_target_pos or (len(self.fixed_tiles) == GRID_SIZE * GRID_SIZE - 1):
                self.all_subgoals_completed = True
                self.status_message = "COMPLETE! Puzzle solved."
            elif current_blank_pos is not None and blank_target_pos is not None:
                self.current_target_tile_and_pos = (0, blank_target_pos)
                self.status_message = f"Subgoal: Move empty tile to {blank_target_pos}"
            else:
                self.all_subgoals_completed = True
                self.status_message = "Completed or no more subgoals."

        if not self.current_target_tile_and_pos and not self.all_subgoals_completed:
            if check_goal_state_standard(self.puzzle_state):
                self.all_subgoals_completed = True
                self.status_message = "COMPLETE! Puzzle solved."
            else:
                self.status_message = "Ready for next subgoal or completed."


    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.solve_button_rect.collidepoint(mouse_pos) and \
                    not self.is_animating and not self.all_subgoals_completed:
                self.solve_current_subgoal()
                return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.reset_simulation()
            return True
        return False

    def solve_current_subgoal(self):
        if self.is_animating or self.all_subgoals_completed or not self.current_target_tile_and_pos:
            if not self.current_target_tile_and_pos and not self.all_subgoals_completed:
                self.determine_next_easiest_subgoal()
                if not self.current_target_tile_and_pos: return
            elif not self.current_target_tile_and_pos and self.all_subgoals_completed:
                return

        tile_to_place, target_pos = self.current_target_tile_and_pos
        if get_tile_pos(self.puzzle_state, tile_to_place) == target_pos:
            self.fixed_tiles[tile_to_place] = target_pos
            self.determine_next_easiest_subgoal()
            return

        self.status_message = f"Finding path for tile {tile_to_place} to {target_pos}..."
        temp_screen = pygame.display.get_surface()
        if temp_screen:
            self.draw(temp_screen, (0, 0), True)
            pygame.display.flip()

        path = a_star_solve_subgoal(self.puzzle_state, tile_to_place, target_pos, self.fixed_tiles)

        if path is not None:
            self.animation_path = path
            self.current_anim_step = 0
            if not path:
                self.fixed_tiles[tile_to_place] = target_pos
                self.determine_next_easiest_subgoal()
            else:
                self.is_animating = True
                self.status_message = f"Moving tile {tile_to_place} ({len(path)} steps)..."
        else:
            self.status_message = f"Error: Path not found for tile {tile_to_place}!"


    def update(self):
        if self.is_animating and self.animation_path:
            current_time = pygame.time.get_ticks()
            if current_time - getattr(self, '_last_anim_time', 0) >= self.animation_delay_per_step:
                self._last_anim_time = current_time
                action = self.animation_path[self.current_anim_step]
                self.puzzle_state = apply_action_standard(self.puzzle_state, action)
                self.current_anim_step += 1
                if self.current_anim_step >= len(self.animation_path):
                    self.is_animating = False
                    self.animation_path = []
                    if self.current_target_tile_and_pos:
                        tile_placed, final_pos = self.current_target_tile_and_pos
                        self.fixed_tiles[tile_placed] = final_pos
                    self.determine_next_easiest_subgoal()
        elif not self.is_animating and not self.all_subgoals_completed:
            if not self.current_target_tile_and_pos:
                self.determine_next_easiest_subgoal()

    def draw(self, screen, mouse_pos, is_drawing_temp_message=False):
        pygame.draw.rect(screen, CONTENT_BACKGROUND_COLOR, self.content_rect)

        btn_text_render = self.solve_button_text_default
        current_btn_bg_color = BUTTON_BG_COLOR
        can_click_solve_btn = not self.is_animating and not self.all_subgoals_completed and self.current_target_tile_and_pos is not None

        if is_drawing_temp_message: can_click_solve_btn = False

        if not can_click_solve_btn:
            current_btn_bg_color = DISABLED_BUTTON_BG_COLOR
            if self.is_animating:
                btn_text_render = "Moving..."
            elif self.all_subgoals_completed:
                btn_text_render = "COMPLETED!"
            elif not self.current_target_tile_and_pos and not self.all_subgoals_completed:
                btn_text_render = "Determining subgoal..."
        elif self.solve_button_rect.collidepoint(mouse_pos):
            current_btn_bg_color = BUTTON_HOVER_COLOR

        pygame.draw.rect(screen, current_btn_bg_color, self.solve_button_rect, border_radius=8)
        solve_text_surf = self.button_font.render(btn_text_render, True, BUTTON_TEXT_COLOR_NEW)
        solve_text_rect = solve_text_surf.get_rect(center=self.solve_button_rect.center)
        screen.blit(solve_text_surf, solve_text_rect)

        status_surf = self.status_font.render(self.status_message, True, TEXT_COLOR_DARK)
        status_rect = status_surf.get_rect(centerx=self.content_rect.centerx, top=self.solve_button_rect.bottom + 20)
        screen.blit(status_surf, status_rect)

        puzzle_start_x = self.content_rect.centerx - self.puzzle_visual_width // 2
        puzzle_start_y = status_rect.bottom + 30
        if self.puzzle_state:
            draw_single_puzzle_and_or(screen, self.tile_font, self.puzzle_state, self.fixed_tiles,
                                      puzzle_start_x, puzzle_start_y, DISPLAY_TILE_SIZE_AND_OR)


# --- Main Application Function ---
def main_app():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH_MAIN, SCREEN_HEIGHT_MAIN))
    pygame.display.set_caption("Group 4: Search in Complex Environments")
    clock = pygame.time.Clock()

    button_font_main = pygame.font.Font(None, 22)

    current_game_state = GameState.PARTIALLY_OBSERVABLE
    content_rect = pygame.Rect(0, CONTENT_Y_OFFSET, SCREEN_WIDTH_MAIN, SCREEN_HEIGHT_MAIN - CONTENT_Y_OFFSET)

    po_simulation = PartiallyObservableSim(content_rect)
    non_obs_auto_random_simulation = NonObservableAutoRandomSim(content_rect)
    and_or_simulation = AndOrSearchSim(content_rect)

    button_height = 36
    button_spacing = 8
    top_margin_header_buttons = (HEADER_HEIGHT - button_height) // 2

    btn_po_text = "Solve All (PO)"
    btn_po_width = button_font_main.size(btn_po_text)[0] + 25
    btn_po_rect = pygame.Rect(10, top_margin_header_buttons, btn_po_width, button_height)

    btn_ao_text = "AND-OR Demo"
    btn_ao_width = button_font_main.size(btn_ao_text)[0] + 25
    btn_ao_rect = pygame.Rect(btn_po_rect.right + button_spacing, top_margin_header_buttons, btn_ao_width,
                              button_height)

    btn_non_obs_text = "Non-Obs (Auto Random)"
    btn_non_obs_width = button_font_main.size(btn_non_obs_text)[0] + 25
    btn_non_obs_rect = pygame.Rect(btn_ao_rect.right + button_spacing, top_margin_header_buttons, btn_non_obs_width,
                                   button_height)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            clicked_on_header_button = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_po_rect.collidepoint(mouse_pos):
                    if current_game_state != GameState.PARTIALLY_OBSERVABLE:
                        po_simulation.reset_simulation_full()
                    current_game_state = GameState.PARTIALLY_OBSERVABLE
                    clicked_on_header_button = True
                elif btn_ao_rect.collidepoint(mouse_pos):
                    if current_game_state != GameState.AND_OR_SEARCH_DEMO:
                        and_or_simulation.reset_simulation()
                    current_game_state = GameState.AND_OR_SEARCH_DEMO
                    clicked_on_header_button = True
                elif btn_non_obs_rect.collidepoint(mouse_pos):
                    if current_game_state != GameState.NON_OBSERVABLE_AUTO_RANDOM:
                        non_obs_auto_random_simulation.reset_simulation()
                    current_game_state = GameState.NON_OBSERVABLE_AUTO_RANDOM
                    clicked_on_header_button = True

            if not clicked_on_header_button:
                if current_game_state == GameState.PARTIALLY_OBSERVABLE:
                    po_simulation.handle_event(event, mouse_pos)
                elif current_game_state == GameState.NON_OBSERVABLE_AUTO_RANDOM:
                    non_obs_auto_random_simulation.handle_event(event, mouse_pos)
                elif current_game_state == GameState.AND_OR_SEARCH_DEMO:
                    and_or_simulation.handle_event(event, mouse_pos)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pass

        if current_game_state == GameState.PARTIALLY_OBSERVABLE:
            po_simulation.update()
        elif current_game_state == GameState.NON_OBSERVABLE_AUTO_RANDOM:
            non_obs_auto_random_simulation.update()
        elif current_game_state == GameState.AND_OR_SEARCH_DEMO:
            and_or_simulation.update()

        screen.fill(APP_BACKGROUND_COLOR)
        pygame.draw.rect(screen, HEADER_BG_COLOR, (0, 0, SCREEN_WIDTH_MAIN, HEADER_HEIGHT))

        header_buttons = [
            (btn_po_rect, btn_po_text, GameState.PARTIALLY_OBSERVABLE),
            (btn_ao_rect, btn_ao_text, GameState.AND_OR_SEARCH_DEMO),
            (btn_non_obs_rect, btn_non_obs_text, GameState.NON_OBSERVABLE_AUTO_RANDOM)
        ]

        for rect, text, state_id in header_buttons:
            color = BUTTON_BG_COLOR
            is_active_tab = (current_game_state == state_id)

            if is_active_tab:
                color = BUTTON_ACTIVE_COLOR
                pygame.draw.rect(screen, HIGHLIGHT_FOUND_GOAL_BORDER,
                                 (rect.left, rect.bottom - 3, rect.width, 3))
            elif rect.collidepoint(mouse_pos):
                color = BUTTON_HOVER_COLOR

            pygame.draw.rect(screen, color, rect, border_radius=6)

            btn_text_surf = button_font_main.render(text, True, BUTTON_TEXT_COLOR_NEW)
            btn_text_rect = btn_text_surf.get_rect(center=rect.center)
            screen.blit(btn_text_surf, btn_text_rect)

        if current_game_state == GameState.PARTIALLY_OBSERVABLE:
            po_simulation.draw(screen, mouse_pos)
        elif current_game_state == GameState.NON_OBSERVABLE_AUTO_RANDOM:
            non_obs_auto_random_simulation.draw(screen, mouse_pos)
        elif current_game_state == GameState.AND_OR_SEARCH_DEMO:
            and_or_simulation.draw(screen, mouse_pos)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# --- Successor Function for A* Subgoal Solving (with fixed_tiles constraint) ---
def get_valid_successors_for_subgoal_a_star(current_state_tuple, fixed_tiles_dict):
    successors = []
    zero_pos = find_zero_pos(current_state_tuple)
    if not zero_pos: return successors
    zero_r, zero_c = zero_pos
    possible_moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    current_state_list = [list(row) for row in current_state_tuple]

    for dr, dc, action_name in possible_moves:
        next_zero_r, next_zero_c = zero_r + dr, zero_c + dc
        if 0 <= next_zero_r < GRID_SIZE and 0 <= next_zero_c < GRID_SIZE:
            tile_to_be_swapped_with_blank = current_state_list[next_zero_r][next_zero_c]
            if tile_to_be_swapped_with_blank in fixed_tiles_dict and \
                    fixed_tiles_dict[tile_to_be_swapped_with_blank] == (next_zero_r, next_zero_c):
                continue

            new_state_list = copy.deepcopy(current_state_list)
            new_state_list[zero_r][zero_c], new_state_list[next_zero_r][next_zero_c] = \
                new_state_list[next_zero_r][next_zero_c], new_state_list[zero_r][zero_c]
            successors.append({'action': action_name, 'state': tuple(map(tuple, new_state_list))})
    return successors


# --- A* Algorithm to Solve Subgoal ---
def a_star_solve_subgoal(initial_state_list, tile_to_place, target_pos, fixed_tiles_dict):
    initial_state_tuple = tuple(map(tuple, initial_state_list))
    current_tile_pos = get_tile_pos(initial_state_tuple, tile_to_place)
    if current_tile_pos == target_pos: return []

    if tile_to_place in fixed_tiles_dict and fixed_tiles_dict[tile_to_place] != target_pos:
        if fixed_tiles_dict[tile_to_place] == current_tile_pos:
            return None

    open_set_entry_count = 0
    h_initial = manhattan_distance(current_tile_pos, target_pos)
    open_set = [(h_initial, 0, open_set_entry_count, initial_state_tuple, [])]
    heapq.heapify(open_set)
    closed_set = {initial_state_tuple: 0}
    max_a_star_nodes = 30000
    nodes_expanded = 0

    while open_set and nodes_expanded < max_a_star_nodes:
        f_cost, g_cost, _entry_c, current_s_tuple, path = heapq.heappop(open_set)
        nodes_expanded += 1
        current_t_pos_iter = get_tile_pos(current_s_tuple, tile_to_place)
        if current_t_pos_iter == target_pos: return path

        for move in get_valid_successors_for_subgoal_a_star(current_s_tuple, fixed_tiles_dict):
            next_s_tuple = move['state']
            action = move['action']
            new_g_cost = g_cost + 1
            if next_s_tuple in closed_set and closed_set[next_s_tuple] <= new_g_cost:
                continue
            closed_set[next_s_tuple] = new_g_cost
            next_t_pos_iter = get_tile_pos(next_s_tuple, tile_to_place)
            if not next_t_pos_iter: continue
            h_cost = manhattan_distance(next_t_pos_iter, target_pos)
            new_f_cost = new_g_cost + h_cost
            open_set_entry_count += 1
            heapq.heappush(open_set, (new_f_cost, new_g_cost, open_set_entry_count, next_s_tuple, path + [action]))
    return None


# --- Draw Function for AND-OR Puzzle ---
def draw_single_puzzle_and_or(screen, font, puzzle_state, fixed_tiles_dict, x_offset, y_offset, tile_size):
    for r_idx, row in enumerate(puzzle_state):
        for c_idx, num in enumerate(row):
            rect = pygame.Rect(x_offset + c_idx * tile_size,
                               y_offset + r_idx * tile_size,
                               tile_size, tile_size)

            is_fixed = (num != 0 and num in fixed_tiles_dict and
                        fixed_tiles_dict[num] == (r_idx, c_idx))

            base_bg_color = TILE_EMPTY_COLOR if num == 0 else TILE_COLOR
            pygame.draw.rect(screen, base_bg_color, rect, border_radius=TILE_BORDER_RADIUS)

            if is_fixed:
                highlight_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
                highlight_surface.fill(HIGHLIGHT_FIXED_TILE)
                screen.blit(highlight_surface, rect.topleft)

            if num != 0:
                text_surf = font.render(str(num), True, TEXT_COLOR_LIGHT)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

            border_w = TILE_BORDER_WIDTH_NEW
            border_c = TILE_BORDER_COLOR_NEW
            if is_fixed:
                border_w = TILE_BORDER_WIDTH_NEW + 1
                border_c = HIGHLIGHT_AND_OR_FIXED_BORDER
            pygame.draw.rect(screen, border_c, rect, border_w, border_radius=TILE_BORDER_RADIUS)


if __name__ == '__main__':
    main_app()
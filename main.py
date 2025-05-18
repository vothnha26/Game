# -*- coding: utf-8 -*-
import sys

import pygame
import heapq
from collections import deque, defaultdict
import copy
from queue import PriorityQueue
import time
import random
import math
import traceback
import numpy as np
import datetime
import pickle
import os
import hashlib
import io
import threading

# Attempt to import matplotlib
try:
    import matplotlib

    matplotlib.use('Agg')  # Use non-interactive backend for generating images
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("CẢNH BÁO: Thư viện matplotlib không được tìm thấy. Chức năng vẽ biểu đồ sẽ bị vô hiệu hóa.")

# --- Hằng số ---
LOG_FILE_NAME = "8_puzzle_test_log.txt"

# Màu sắc
WINDOW_BG_COLOR = (240, 240, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 128, 255)
TILE_BORDER_COLOR = BLACK
EMPTY_TILE_BG_COLOR = (200, 200, 200)
LIGHT_BLUE = (100, 149, 237)
STEEL_BLUE = (80, 140, 190)
ORANGE = (255, 140, 0)
GREEN = (0, 150, 0)  # For Run button and comparison buttons
MEDIUM_SEA_GREEN = (60, 179, 113)  # For comparison buttons
RED = (200, 0, 0)
TEXT_COLOR = WHITE
INFO_TEXT_COLOR = BLACK
TITLE_COLOR = (50, 50, 50)
PATH_TEXT_COLOR = (30, 30, 30)
PATH_BG_COLOR = (225, 225, 225)
SCROLLBAR_BG_COLOR = (200, 200, 200)
SCROLLBAR_HANDLE_COLOR = (130, 130, 130)
SCROLLBAR_HANDLE_HOVER_COLOR = (100, 100, 100)

# --- Tham số Thuật toán Di truyền (GA) ---
GA_POPULATION_SIZE = 100
GA_MAX_GENERATIONS = 150
GA_ELITISM_COUNT = 5
GA_TOURNAMENT_SIZE = 5
GA_MUTATION_RATE = 0.15
GA_MAX_SEQUENCE_LENGTH = 60  # Max length of a move sequence for an individual

# --- Tham số Q-Learning ---
QL_ALPHA = 0.1
QL_GAMMA = 0.9
QL_EPSILON_START = 1.0
QL_EPSILON_END = 0.01
QL_EPSILON_DECAY = 0.995  # Giữ nguyên decay này, tăng episodes sẽ cho epsilon đủ thời gian giảm
QL_EPISODES = 20000
QL_MAX_STEPS_PER_EPISODE = 150
QL_REWARD_GOAL = 100
QL_REWARD_MOVE = -1
QL_REWARD_PREVIOUS = -10

# --- Tham số So sánh Thuật toán ---
MAX_UNINFORMED_ALGO_RUNTIME = 30
MAX_LOCAL_SEARCH_ALGO_RUNTIME = 45

# Kích thước màn hình và lưới
SCREEN_WIDTH = 1350
SCREEN_HEIGHT = 750
GRID_SIZE = 3
TILE_SIZE = 100
TILE_BORDER_WIDTH = 1
PADDING = 15

# Vị trí các bàn cờ
BOARD_TOTAL_WIDTH = GRID_SIZE * TILE_SIZE
BOARD_TOTAL_HEIGHT = BOARD_TOTAL_WIDTH
BOARD_OFFSET_X_START = PADDING * 3
BOARD_OFFSET_Y = PADDING * 5
BOARD_SPACING_X = PADDING * 4
BOARD_OFFSET_X_GOAL = BOARD_OFFSET_X_START + BOARD_TOTAL_WIDTH + BOARD_SPACING_X

# Kích thước và vị trí nút bấm
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_PADDING = 8
BUTTON_BORDER_RADIUS = 10
BUTTON_START_X_COL1 = BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH + PADDING * 4
COL_SPACING = PADDING * 2
BUTTON_START_X_COL2 = BUTTON_START_X_COL1 + BUTTON_WIDTH + COL_SPACING
BUTTON_COL_START_Y = BOARD_OFFSET_Y

# --- Khu vực Hiển thị Đường đi & Thanh cuộn ---
SCROLLBAR_WIDTH = 15
PATH_AREA_X_START = BUTTON_START_X_COL2 + BUTTON_WIDTH + PADDING * 4
PATH_AREA_Y_START = BOARD_OFFSET_Y
PATH_AREA_WIDTH = SCREEN_WIDTH - PATH_AREA_X_START - PADDING * 2 - SCROLLBAR_WIDTH - PADDING // 2
PATH_AREA_HEIGHT = 650 - PATH_AREA_Y_START - PADDING * 2
PATH_LINE_SPACING = 5
SCROLLBAR_X = PATH_AREA_X_START + PATH_AREA_WIDTH + PADDING // 2
SCROLLBAR_Y = PATH_AREA_Y_START
SCROLLBAR_HEIGHT = PATH_AREA_HEIGHT

# Font chữ
FONT_NAME = "Arial"
FONT_SIZE_TILE = 40
FONT_SIZE_BUTTON = 15
FONT_SIZE_INFO = 18
FONT_SIZE_TITLE = 22
FONT_SIZE_PATH = 14

# --- Hằng số Cụ thể cho Thuật toán ---
BEAM_WIDTH = 10

# --- Hằng số Hoạt ảnh ---
MAX_ANIMATION_STEPS_BEFORE_ACCELERATION = 1000
NORMAL_ANIMATION_DELAY = 250
FAST_ANIMATION_DELAY = 10

# --- Trạng thái Mặc định của Trò chơi ---
begin_state = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# --- Biểu đồ ---
CHART_WIDTH = (BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH) - BOARD_OFFSET_X_START - PADDING
CHART_HEIGHT = 220


# --- Lớp Trợ giúp cho việc chạy Thuật toán trong Thread ---
class AlgorithmRunResult:
    def __init__(self):
        self.solution = None
        self.error_message = None
        self.exception_occurred = False


def run_algorithm_in_thread_wrapper(solver_func, initial_state_c, goal_state_c, result_obj):
    try:
        copied_initial = copy.deepcopy(initial_state_c)
        copied_goal = copy.deepcopy(goal_state_c)
        result_obj.solution = solver_func(copied_initial, copied_goal)
    except MemoryError as e_mem:
        result_obj.error_message = f"Lỗi Bộ Nhớ trong luồng: {type(e_mem).__name__}: {str(e_mem)[:150]}"
        result_obj.exception_occurred = True
        print(f"--- Lỗi Bộ Nhớ trong Luồng cho {solver_func.__name__} ---")
        traceback.print_exc()
        print(f"--- Kết thúc Chi tiết Lỗi Bộ Nhớ ---")
    except Exception as e:
        result_obj.error_message = f"Lỗi Runtime trong luồng: {type(e).__name__}: {str(e)[:150]}"
        result_obj.exception_occurred = True
        print(f"--- Chi tiết Lỗi Runtime trong Luồng cho {solver_func.__name__} ---")
        traceback.print_exc()
        print(f"--- Kết thúc Chi tiết Lỗi Runtime ---")


# --- Hàm trợ giúp ---
def find_zero_pos(state):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state[r][c] == 0:
                return r, c
    return -1, -1


def find_tile_pos(num, target_state):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if target_state[r][c] == num:
                return r, c
    return -1, -1


def state_to_tuple(state):
    return tuple(map(tuple, state))


def manhattan_distance(state, target_state):
    dist = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            num = state[r][c]
            if num != 0:
                goal_r, goal_c = find_tile_pos(num, target_state)
                if goal_r != -1:
                    dist += abs(r - goal_r) + abs(c - goal_c)
    return dist


def get_neighbors(state):
    neighbors = []
    zero_r, zero_c = find_zero_pos(state)
    if zero_r == -1: return []
    for dr, dc in moves:
        next_r, next_c = zero_r + dr, zero_c + dc
        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[zero_r][zero_c], new_state[next_r][next_c] = \
                new_state[next_r][next_c], new_state[zero_r][zero_c]
            neighbors.append(new_state)
    return neighbors


def get_state_hash(state_tuple):
    return hashlib.md5(str(state_tuple).encode()).hexdigest()[:16]


# --- Các hàm GA, Thuật toán tìm kiếm, Q-Learning ---

# Genetic Algorithm Helper Functions
def apply_move_sequence(start_state, move_sequence):
    current_state = copy.deepcopy(start_state)
    path = [current_state]
    possible_moves_map = {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}
    reverse_moves_map = {v: k for k, v in possible_moves_map.items()}

    for move_str in move_sequence:
        zero_r, zero_c = find_zero_pos(current_state)
        if zero_r == -1: break

        move_delta = reverse_moves_map.get(move_str)
        if move_delta is None: continue

        dr, dc = move_delta
        next_r, next_c = zero_r + dr, zero_c + dc

        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            new_state_ga = copy.deepcopy(current_state)
            new_state_ga[zero_r][zero_c], new_state_ga[next_r][next_c] = new_state_ga[next_r][next_c], \
            new_state_ga[zero_r][zero_c]
            current_state = new_state_ga
            path.append(current_state)
    return current_state, path


def calculate_fitness(state, target_state):
    return manhattan_distance(state, target_state)


def create_random_individual():
    return [random.choice(action_names) for _ in range(GA_MAX_SEQUENCE_LENGTH)]


def tournament_selection(population_with_fitness, k):
    selected_tournament = random.sample(population_with_fitness, k)
    selected_tournament.sort(key=lambda x: x[1])
    return selected_tournament[0][0]


def single_point_crossover(parent1, parent2):
    if not isinstance(parent1, list) or not isinstance(parent2, list) or len(parent1) < 2 or len(parent2) < 2:
        return parent1[:], parent2[:]
    min_len = min(len(parent1), len(parent2))
    if min_len < 2: return parent1[:], parent2[:]
    point = random.randint(1, min_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate_sequence(sequence, mutation_rate):
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            current_move = mutated_sequence[i]
            possible_new_moves = [m for m in action_names if m != current_move]
            new_move = random.choice(possible_new_moves if possible_new_moves else action_names)
            mutated_sequence[i] = new_move
    return mutated_sequence


def GeneticAlgorithm(initial_state, target_state):
    print(
        f"Đang chạy Genetic Algorithm: Pop={GA_POPULATION_SIZE}, Gens={GA_MAX_GENERATIONS}, SeqLen={GA_MAX_SEQUENCE_LENGTH}, MutRate={GA_MUTATION_RATE}")
    start_timer = time.time()
    population = [create_random_individual() for _ in range(GA_POPULATION_SIZE)]
    best_overall_individual = None
    best_overall_fitness = float('inf')

    for generation in range(GA_MAX_GENERATIONS):
        population_with_fitness = []
        goal_found_this_gen = False
        solution_path_this_gen = None

        for individual_sequence in population:
            final_state_of_seq, _ = apply_move_sequence(initial_state, individual_sequence)
            fitness = calculate_fitness(final_state_of_seq, target_state)
            population_with_fitness.append((individual_sequence, fitness))

            if fitness == 0:
                print(f"GA tìm thấy trạng thái đích ở thế hệ {generation}!")
                _, solution_path_this_gen = apply_move_sequence(initial_state, individual_sequence)
                goal_found_this_gen = True
                break

        if goal_found_this_gen:
            end_timer = time.time()
            print(f"Thời gian GA: {end_timer - start_timer:.4f} giây")
            return solution_path_this_gen

        population_with_fitness.sort(key=lambda x: x[1])

        current_best_individual_this_gen, current_best_fitness_this_gen = population_with_fitness[0]
        if current_best_fitness_this_gen < best_overall_fitness:
            best_overall_fitness = current_best_fitness_this_gen
            best_overall_individual = current_best_individual_this_gen

        if generation % 30 == 0 or generation == GA_MAX_GENERATIONS - 1:
            print(
                f"Thế hệ {generation}: Fitness tốt nhất hiện tại={current_best_fitness_this_gen}, Tổng thể tốt nhất={best_overall_fitness}")

        new_population = []
        elites = [ind for ind, fit in population_with_fitness[:GA_ELITISM_COUNT]]
        new_population.extend(elites)

        while len(new_population) < GA_POPULATION_SIZE:
            parent1 = tournament_selection(population_with_fitness, GA_TOURNAMENT_SIZE)
            parent2 = tournament_selection(population_with_fitness, GA_TOURNAMENT_SIZE)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate_sequence(child1, GA_MUTATION_RATE)
            child2 = mutate_sequence(child2, GA_MUTATION_RATE)
            new_population.append(child1)
            if len(new_population) < GA_POPULATION_SIZE:
                new_population.append(child2)
        population = new_population

    end_timer = time.time()
    print(
        f"GA kết thúc sau {GA_MAX_GENERATIONS} thế hệ. Fitness tốt nhất: {best_overall_fitness}. Thời gian GA: {end_timer - start_timer:.4f} giây")

    if best_overall_individual:
        final_state_of_best, path_of_best = apply_move_sequence(initial_state, best_overall_individual)
        final_fitness = calculate_fitness(final_state_of_best, target_state)
        print(
            f"Trả về đường đi của cá thể tốt nhất (dài {len(path_of_best) - 1} bước thực tế). Fitness cuối cùng của cá thể: {final_fitness}")
        return path_of_best
    else:
        print("GA: Không tìm thấy cá thể phù hợp nào. Trả về trạng thái ban đầu.")
        return [initial_state]


# --- Search Algorithms ---
def BFS(start_state, target_state):
    print("Đang chạy BFS...")
    queue = deque([(start_state, [start_state])])
    visited = {state_to_tuple(start_state)}
    while queue:
        current_state, path = queue.popleft()
        if current_state == target_state:
            return path
        for neighbor_state in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor_state)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor_state, path + [neighbor_state]))
    print("BFS: Không tìm thấy lời giải.")
    return None


def UCS(initial_state, target_state):
    print("Đang chạy UCS...")
    pq = [(0, initial_state, [initial_state])]
    visited = {state_to_tuple(initial_state): 0}

    while pq:
        cost, current_state, path = heapq.heappop(pq)

        if current_state == target_state:
            return path

        current_state_tuple = state_to_tuple(current_state)
        if cost > visited.get(current_state_tuple, float('inf')):
            continue

        for neighbor_state in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor_state)
            new_cost = cost + 1

            if new_cost < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_cost
                heapq.heappush(pq, (new_cost, neighbor_state, path + [neighbor_state]))
    print("UCS: Không tìm thấy lời giải.")
    return None


def DFS(initial_state, target_state):
    print("Đang chạy DFS...")
    max_practical_dfs_depth = 35
    stack = [(initial_state, [initial_state], 0)]
    visited = {state_to_tuple(initial_state)}

    while stack:
        current_state, path, depth = stack.pop()

        if current_state == target_state:
            return path

        if depth >= max_practical_dfs_depth:
            continue

        for neighbor_state in reversed(get_neighbors(current_state)):
            neighbor_tuple = state_to_tuple(neighbor_state)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                stack.append((neighbor_state, path + [neighbor_state], depth + 1))
    print(f"DFS: Không tìm thấy lời giải hoặc đạt đến độ sâu tối đa ({max_practical_dfs_depth}).")
    return None


def depth_limited_dfs(start_state, target_state, depth_limit):
    stack = [(start_state, [start_state], 0)]
    visited_depth = {state_to_tuple(start_state): 0}

    while stack:
        current_state, path, depth = stack.pop()

        if current_state == target_state:
            return path

        if depth < depth_limit:
            for neighbor_state in reversed(get_neighbors(current_state)):
                neighbor_tuple = state_to_tuple(neighbor_state)
                new_depth = depth + 1
                if new_depth < visited_depth.get(neighbor_tuple, float('inf')):
                    visited_depth[neighbor_tuple] = new_depth
                    stack.append((neighbor_state, path + [neighbor_state], new_depth))
    return None


def IDDFS(initial_state, target_state):
    print("Đang chạy IDDFS...")
    depth_limit = 0
    max_iddfs_depth = 31
    while True:
        print(f"IDDFS đang thử độ sâu: {depth_limit}")
        solution_path = depth_limited_dfs(initial_state, target_state, depth_limit)
        if solution_path is not None:
            return solution_path
        depth_limit += 1
        if depth_limit > max_iddfs_depth:
            print(f"IDDFS đạt đến độ sâu thực tế tối đa ({max_iddfs_depth}) mà không có lời giải.")
            return None


def A_Star(initial_state, target_state):
    print("Đang chạy A*...")
    pq = PriorityQueue()
    start_h = manhattan_distance(initial_state, target_state)
    pq.put((start_h, 0, initial_state, [initial_state]))
    visited_cost = {state_to_tuple(initial_state): 0}

    while not pq.empty():
        f_n, g_cost, current_state, path = pq.get()

        if current_state == target_state:
            return path

        current_state_tuple = state_to_tuple(current_state)
        if g_cost > visited_cost.get(current_state_tuple, float('inf')):
            continue

        for neighbor_state in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor_state)
            new_g_cost = g_cost + 1

            if new_g_cost < visited_cost.get(neighbor_tuple, float('inf')):
                visited_cost[neighbor_tuple] = new_g_cost
                h_cost = manhattan_distance(neighbor_state, target_state)
                f_cost = new_g_cost + h_cost
                pq.put((f_cost, new_g_cost, neighbor_state, path + [neighbor_state]))
    print("A*: Không tìm thấy lời giải.")
    return None


def Greedy(initial_state, target_state):
    print("Đang chạy Greedy Best-First...")
    pq = PriorityQueue()
    start_h = manhattan_distance(initial_state, target_state)
    pq.put((start_h, initial_state, [initial_state]))
    visited = {state_to_tuple(initial_state)}

    while not pq.empty():
        h_n, current_state, path = pq.get()

        if current_state == target_state:
            return path

        for neighbor_state in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor_state)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                h_cost = manhattan_distance(neighbor_state, target_state)
                pq.put((h_cost, neighbor_state, path + [neighbor_state]))
    print("Greedy: Không tìm thấy lời giải.")
    return None


def depth_limited_A_star(start_state, target_state, f_limit):
    min_exceeded_f = float('inf')
    _stack = [(0, start_state, [start_state])]
    visited_in_pass = {state_to_tuple(start_state): 0}

    while _stack:
        g_cost, current_state, path = _stack.pop()
        current_f = g_cost + manhattan_distance(current_state, target_state)

        if current_f > f_limit:
            min_exceeded_f = min(min_exceeded_f, current_f)
            continue

        if current_state == target_state:
            return path, float('inf')

        for neighbor_state in reversed(get_neighbors(current_state)):
            new_g_cost = g_cost + 1
            neighbor_tuple = state_to_tuple(neighbor_state)

            if new_g_cost >= visited_in_pass.get(neighbor_tuple, float('inf')):
                continue

            neighbor_f = new_g_cost + manhattan_distance(neighbor_state, target_state)

            if neighbor_f <= f_limit:
                visited_in_pass[neighbor_tuple] = new_g_cost
                _stack.append((new_g_cost, neighbor_state, path + [neighbor_state]))
            else:
                min_exceeded_f = min(min_exceeded_f, neighbor_f)
    return None, min_exceeded_f


def IDA(initial_state, target_state):
    print("Đang chạy IDA*...")
    f_limit = manhattan_distance(initial_state, target_state)
    max_f_limit_practical = 80

    while True:
        print(f"IDA* đang thử f-limit: {f_limit}")
        solution_path_ida, next_f_limit_candidate = depth_limited_A_star(initial_state, target_state, f_limit)

        if solution_path_ida is not None:
            return solution_path_ida

        if next_f_limit_candidate == float('inf'):
            print("IDA*: Không tìm thấy lời giải (đã khám phá tất cả các nút có thể).")
            return None

        if next_f_limit_candidate > max_f_limit_practical:
            print(
                f"IDA* f-limit tiếp theo ({next_f_limit_candidate}) vượt quá ngưỡng thực tế ({max_f_limit_practical}). Đang dừng.")
            return None

        if next_f_limit_candidate <= f_limit:
            print(
                f"IDA* f-limit không tăng ({next_f_limit_candidate} <= {f_limit}). Có thể bị kẹt hoặc lỗi. Đang dừng.")
            return None
        f_limit = next_f_limit_candidate


# Local Search Algorithms
def SimpleHillClimbing(initial_state, target_state):
    print("Đang chạy Simple Hill Climbing...")
    current_state = copy.deepcopy(initial_state)
    path = [current_state]
    max_steps_hc = 1000

    for _ in range(max_steps_hc):
        if current_state == target_state: return path
        current_h = manhattan_distance(current_state, target_state)
        if current_h == 0: return path

        best_neighbor_simple = None
        found_better_simple = False
        neighbors_simple = get_neighbors(current_state)
        random.shuffle(neighbors_simple)

        for neighbor in neighbors_simple:
            neighbor_h = manhattan_distance(neighbor, target_state)
            if neighbor_h < current_h:
                best_neighbor_simple = neighbor
                found_better_simple = True
                break

        if not found_better_simple:
            print("Simple HC: Đạt đến cực đại địa phương hoặc vùng bình nguyên.")
            return path

        current_state = best_neighbor_simple
        path.append(current_state)

    print(f"Simple HC: Đạt số bước tối đa ({max_steps_hc}).")
    return path


def SteepestHillClimbing(initial_state, target_state):
    print("Đang chạy Steepest Ascent Hill Climbing...")
    current_state = copy.deepcopy(initial_state)
    path = [current_state]
    max_steps_hc_steep = 1000

    for _ in range(max_steps_hc_steep):
        if current_state == target_state: return path
        current_h = manhattan_distance(current_state, target_state)
        if current_h == 0: return path

        best_neighbor_steep = None
        best_h_steep = current_h
        neighbors_steep = get_neighbors(current_state)

        for neighbor in neighbors_steep:
            neighbor_h = manhattan_distance(neighbor, target_state)
            if neighbor_h < best_h_steep:
                best_h_steep = neighbor_h
                best_neighbor_steep = neighbor

        if best_neighbor_steep is None:
            print("Steepest HC: Đạt đến cực đại địa phương hoặc vùng bình nguyên.")
            return path

        current_state = best_neighbor_steep
        path.append(current_state)

    print(f"Steepest HC: Đạt số bước tối đa ({max_steps_hc_steep}).")
    return path


def StochasticHillClimbing(initial_state, target_state):
    print("Đang chạy Stochastic Hill Climbing...")
    current_state = copy.deepcopy(initial_state)
    path = [current_state]
    max_steps_hc_stoch = 1000

    for _ in range(max_steps_hc_stoch):
        if current_state == target_state: return path
        current_h = manhattan_distance(current_state, target_state)
        if current_h == 0: return path

        uphill_neighbors_stoch = []
        neighbors_stoch = get_neighbors(current_state)
        for neighbor in neighbors_stoch:
            neighbor_h = manhattan_distance(neighbor, target_state)
            if neighbor_h < current_h:
                uphill_neighbors_stoch.append(neighbor)

        if not uphill_neighbors_stoch:
            print("Stochastic HC: Đạt đến cực đại địa phương (không có nước đi lên dốc).")
            return path

        current_state = random.choice(uphill_neighbors_stoch)
        path.append(current_state)

    print(f"Stochastic HC: Đạt số bước tối đa ({max_steps_hc_stoch}).")
    return path


def SimulatedAnnealing(initial_state, target_state):
    print("Đang chạy Simulated Annealing...")
    temp_start = 15.0
    temp_end = 0.01
    alpha = 0.98
    max_iter_per_temp = 100

    current_state = copy.deepcopy(initial_state)
    current_cost = manhattan_distance(current_state, target_state)
    path = [current_state]
    best_state_overall = current_state
    best_cost_overall = current_cost
    temp = temp_start

    while temp > temp_end:
        iter_count = 0
        while iter_count < max_iter_per_temp:
            iter_count += 1

            if current_state == target_state:
                print(f"SA đạt trạng thái đích ở nhiệt độ {temp:.3f}")
                if not path or state_to_tuple(path[-1]) != state_to_tuple(current_state):
                    path.append(copy.deepcopy(current_state))
                return path

            neighbors = get_neighbors(current_state)
            if not neighbors: continue

            neighbor_state = random.choice(neighbors)
            neighbor_cost = manhattan_distance(neighbor_state, target_state)
            delta_e = neighbor_cost - current_cost
            accept_move = False

            if delta_e < 0:
                accept_move = True
                if neighbor_cost < best_cost_overall:
                    best_state_overall = copy.deepcopy(neighbor_state)
                    best_cost_overall = neighbor_cost
            elif temp > 1e-9:
                acceptance_probability = math.exp(-delta_e / temp)
                if random.random() < acceptance_probability:
                    accept_move = True

            if accept_move:
                current_state = neighbor_state
                current_cost = neighbor_cost
                if not path or state_to_tuple(path[-1]) != state_to_tuple(current_state):
                    path.append(copy.deepcopy(current_state))
        temp *= alpha

    print(f"SA kết thúc. Chi phí cuối cùng: {current_cost}, Chi phí tốt nhất tìm thấy: {best_cost_overall}")
    if not path or state_to_tuple(path[-1]) != state_to_tuple(current_state):
        path.append(copy.deepcopy(current_state))

    if best_cost_overall == 0 and current_cost != 0:
        print("SA Cảnh báo: Đã thăm trạng thái đích nhưng trạng thái cuối cùng khác. Trả về đường đi khám phá thực tế.")
    return path


def BeamSearch(initial_state, target_state, beam_width):
    print(f"Đang chạy Beam Search (Độ rộng={beam_width})...")
    start_h = manhattan_distance(initial_state, target_state)
    beam = [(start_h, initial_state, [initial_state])]
    visited_beam_tuples = {state_to_tuple(initial_state)}
    max_iterations_beam = 500
    iterations = 0

    while beam and iterations < max_iterations_beam:
        iterations += 1
        possible_next_candidates = []
        goal_found_in_beam_iteration = False
        solution_path_beam_iteration = None

        for h_val, current_beam_state, current_path in beam:
            if current_beam_state == target_state:
                print(f"Beam Search tìm thấy lời giải trong chùm ở lần lặp {iterations}.")
                if solution_path_beam_iteration is None or len(current_path) < len(solution_path_beam_iteration):
                    solution_path_beam_iteration = current_path
                goal_found_in_beam_iteration = True

            for neighbor in get_neighbors(current_beam_state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited_beam_tuples or neighbor == target_state:
                    neighbor_h = manhattan_distance(neighbor, target_state)
                    possible_next_candidates.append((neighbor_h, neighbor, current_path + [neighbor]))

        if goal_found_in_beam_iteration and solution_path_beam_iteration:
            print(f"Beam Search: Trả về lời giải tìm thấy.")
            return solution_path_beam_iteration

        if not possible_next_candidates:
            print(f"Beam Search: Chùm trống sau {iterations} lần lặp. Không có ứng viên mới.")
            beam.sort(key=lambda x: x[0])
            return beam[0][2] if beam else [initial_state]

        possible_next_candidates.sort(key=lambda x: x[0])
        new_beam = []
        temp_tuples_for_new_beam = set()
        for cand_h, cand_state, cand_path in possible_next_candidates:
            if len(new_beam) < beam_width:
                cand_tuple = state_to_tuple(cand_state)
                if cand_tuple not in temp_tuples_for_new_beam:
                    new_beam.append((cand_h, cand_state, cand_path))
                    temp_tuples_for_new_beam.add(cand_tuple)
            else:
                break
        beam = new_beam
        visited_beam_tuples.update(temp_tuples_for_new_beam)

        if not beam:
            print("Beam Search: Chùm tìm kiếm mới trống. Không tìm thấy lời giải.")
            return [initial_state]

    if iterations >= max_iterations_beam:
        print(f"Beam Search đạt số lần lặp tối đa ({max_iterations_beam}).")

    if beam:
        beam.sort(key=lambda x: x[0])
        print(f"Beam Search không tìm thấy lời giải. Trả về đường đi tốt nhất từ chùm cuối (h={beam[0][0]}).")
        return beam[0][2]
    else:
        print("Beam Search không tìm thấy lời giải và chùm cuối trống.")
        return [initial_state]


# Q-Learning
def default_q_value_factory():
    return np.zeros(len(moves))


def QLearning(initial_state, target_state): # Renamed q_study to QLearning to match existing call
    print(
        f"Đang chạy Q-Learning (q_study logic): Episodes={QL_EPISODES}, MaxSteps/Ep={QL_MAX_STEPS_PER_EPISODE}, Alpha={QL_ALPHA}, Gamma={QL_GAMMA}, EpsilonDecay={QL_EPSILON_DECAY}")

    initial_state_tuple = state_to_tuple(initial_state)
    target_state_tuple = state_to_tuple(target_state)
    q_table_filename = f"q_table_goal_{get_state_hash(target_state_tuple)}_qstudy.pkl" # Modified filename slightly
    q_table_loaded_successfully = False
    q_table_trained_this_session = False

    # In q_study, q_table keys are (empty_tile_row, empty_tile_col)
    # We'll adapt this to use full state tuples as keys for consistency with other parts of your Pygame code,
    # but the core logic of q_study (mapping empty tile pos to action values) will be approximated.
    # A more direct translation of q_study's Q-table would be:
    # q_table = defaultdict(lambda: np.zeros(len(moves)))
    # And keys would be find_zero_pos(state)
    # For now, let's stick to state_tuple keys and adapt.

    q_table = defaultdict(default_q_value_factory) # Q-values for (state_tuple, action_index)

    if os.path.exists(q_table_filename):
        try:
            with open(q_table_filename, 'rb') as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, defaultdict) and loaded_data.default_factory == default_q_value_factory:
                    q_table = loaded_data
                    print(f"Đã tải Q-table (q_study logic) từ file: {q_table_filename}.")
                    q_table_loaded_successfully = True
                else:
                    print(f"Định dạng Q-table trong {q_table_filename} không nhận dạng được. Training lại.")
        except Exception as e:
            print(f"Lỗi khi tải Q-table (q_study logic): {e}. Sẽ training lại từ đầu.")
            q_table = defaultdict(default_q_value_factory)

    current_epsilon = QL_EPSILON_START

    if not q_table_loaded_successfully:
        print("Bắt đầu giai đoạn training Q-Learning (q_study logic)...")
        q_table_trained_this_session = True

        for episode in range(QL_EPISODES):
            current_s_list_ql = copy.deepcopy(initial_state)
            # Path tracking for the episode, not strictly needed for q_study's core learning but can be useful
            # episode_path = [current_s_list_ql]

            for step in range(QL_MAX_STEPS_PER_EPISODE):
                current_s_tuple_ql = state_to_tuple(current_s_list_ql)
                zero_r_ql, zero_c_ql = find_zero_pos(current_s_list_ql) # Your Pygame's find_zero_pos

                if zero_r_ql == -1: # Should not happen in 8-puzzle
                    break

                action_idx = -1
                # Epsilon-greedy action selection
                if random.random() < current_epsilon:
                    # Get valid actions from current empty tile position
                    valid_actions_indices_episode = []
                    for i_act, (dr_act_check, dc_act_check) in enumerate(moves):
                        if 0 <= zero_r_ql + dr_act_check < GRID_SIZE and 0 <= zero_c_ql + dc_act_check < GRID_SIZE:
                            valid_actions_indices_episode.append(i_act)
                    if valid_actions_indices_episode:
                        action_idx = random.choice(valid_actions_indices_episode)
                    else:
                        break # No valid moves
                else:
                    # Exploit: choose the action with the highest Q-value
                    # For q_study's original Q-table structure (keyed by empty_pos):
                    # q_values_for_empty_pos = q_table[(zero_r_ql, zero_c_ql)]
                    # For current structure (keyed by state_tuple):
                    q_values_for_empty_pos = q_table[current_s_tuple_ql]

                    best_q_val_episode = -float('inf')
                    candidate_actions_episode = []
                    for i_act, (dr_act_check, dc_act_check) in enumerate(moves):
                         if 0 <= zero_r_ql + dr_act_check < GRID_SIZE and 0 <= zero_c_ql + dc_act_check < GRID_SIZE:
                            if q_values_for_empty_pos[i_act] > best_q_val_episode:
                                best_q_val_episode = q_values_for_empty_pos[i_act]
                                candidate_actions_episode = [i_act]
                            elif q_values_for_empty_pos[i_act] == best_q_val_episode:
                                candidate_actions_episode.append(i_act)
                    if candidate_actions_episode:
                        action_idx = random.choice(candidate_actions_episode)
                    else: # No valid/learned actions, pick random valid
                        valid_actions_indices_episode_fallback = []
                        for i_act_fb, (dr_act_fb, dc_act_fb) in enumerate(moves):
                            if 0 <= zero_r_ql + dr_act_fb < GRID_SIZE and 0 <= zero_c_ql + dc_act_fb < GRID_SIZE:
                                valid_actions_indices_episode_fallback.append(i_act_fb)
                        if valid_actions_indices_episode_fallback:
                            action_idx = random.choice(valid_actions_indices_episode_fallback)
                        else:
                            break


                if action_idx == -1: # Should not happen if logic above is correct
                    break

                # Perform the action
                dr_act, dc_act = moves[action_idx]
                next_s_list_ql = copy.deepcopy(current_s_list_ql)
                # Apply move: Chinh_Sua_Ma_Tran equivalent
                # Your Pygame code uses get_neighbors, let's simulate one step
                _zr_ql_act, _zc_ql_act = find_zero_pos(next_s_list_ql) # Re-find, though it's zero_r_ql, zero_c_ql
                next_r_moved, next_c_moved = _zr_ql_act + dr_act, _zc_ql_act + dc_act

                # Check if this move is valid (it should be if action_idx was chosen from valid ones)
                if not (0 <= next_r_moved < GRID_SIZE and 0 <= next_c_moved < GRID_SIZE):
                    # This case implies an issue with action selection if it occurs
                    # Potentially add a penalty or skip update
                    print(f"Warning: Invalid move selected in QLearning episode. Pos:({_zr_ql_act},{_zc_ql_act}), Action: {action_idx} -> ({next_r_moved},{next_c_moved})")
                    # For robustness, we might skip this step or penalize
                    # For now, let's assume valid_actions_indices ensures this doesn't happen often
                    continue


                next_s_list_ql[_zr_ql_act][_zc_ql_act], next_s_list_ql[next_r_moved][next_c_moved] = \
                    next_s_list_ql[next_r_moved][next_c_moved], next_s_list_ql[_zr_ql_act][_zc_ql_act]
                next_s_tuple_ql = state_to_tuple(next_s_list_ql)
                # episode_path.append(next_s_list_ql) # If tracking episode path

                # Define reward
                reward = QL_REWARD_MOVE
                if next_s_tuple_ql == target_state_tuple:
                    reward = QL_REWARD_GOAL
                # The q_study had `reward + max(q_table[(new_x,new_y)])`
                # Adapting: `max(q_table[next_s_tuple_ql])`
                # And your original QL has alpha and gamma.

                old_q_value = q_table[current_s_tuple_ql][action_idx] # Q(s,a)

                # Max Q-value for the next state Q(s', a')
                max_future_q = 0
                if next_s_tuple_ql != target_state_tuple: # If next_state is not terminal
                    q_values_for_next_state_options = q_table[next_s_tuple_ql]
                    # Consider only valid moves from next_state for max_future_q
                    next_zero_r_ql, next_zero_c_ql = find_zero_pos(next_s_list_ql)
                    possible_future_q_values = []
                    if next_zero_r_ql != -1:
                        for i_future_act, (dr_future, dc_future) in enumerate(moves):
                            if 0 <= next_zero_r_ql + dr_future < GRID_SIZE and \
                               0 <= next_zero_c_ql + dc_future < GRID_SIZE:
                                possible_future_q_values.append(q_values_for_next_state_options[i_future_act])
                    if possible_future_q_values:
                         max_future_q = np.max(possible_future_q_values)
                    # else: max_future_q remains 0, e.g. if next_state has no valid moves (shouldn't happen)


                # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a'(Q(s',a')) - Q(s,a))
                new_q_value = old_q_value + QL_ALPHA * (reward + QL_GAMMA * max_future_q - old_q_value)
                q_table[current_s_tuple_ql][action_idx] = new_q_value

                current_s_list_ql = next_s_list_ql # Move to the new state

                if reward == QL_REWARD_GOAL:
                    break # End episode if goal is reached

            current_epsilon = max(QL_EPSILON_END, current_epsilon * QL_EPSILON_DECAY)
            if episode % (QL_EPISODES // 20) == 0 or episode == QL_EPISODES - 1:
                print(
                    f"Q-Learning Training (q_study logic): Episode {episode + 1}/{QL_EPISODES}, Epsilon: {current_epsilon:.4f}, Q-Table size: {len(q_table)}")
        print("Hoàn thành giai đoạn training Q-Learning (q_study logic).")
    else:
        current_epsilon = QL_EPSILON_END # If loaded, use final epsilon for reconstruction
        print("Q-table (q_study logic) đã được tải, bỏ qua training, sẵn sàng tái tạo đường đi.")


    # Path Reconstruction (Greedy policy based on learned Q-table)
    print("Q-Learning (q_study logic): Bắt đầu tái tạo đường đi từ Q-table...")
    path_recon = [initial_state]
    current_s_list_recon = copy.deepcopy(initial_state)
    path_found_to_goal_ql = False
    visited_recon_states = {state_to_tuple(current_s_list_recon)}
    max_reconstruction_steps = QL_MAX_STEPS_PER_EPISODE * 2 # Allow more steps for reconstruction

    for step_recon in range(max_reconstruction_steps):
        current_s_tuple_recon = state_to_tuple(current_s_list_recon)
        if current_s_tuple_recon == target_state_tuple:
            print("Q-Learning (q_study logic): Đã tìm thấy đường đi tới đích từ Q-table.")
            path_found_to_goal_ql = True
            break

        q_values_recon = q_table[current_s_tuple_recon] # Get Q-values for current state
        zero_r_recon, zero_c_recon = find_zero_pos(current_s_list_recon)

        if zero_r_recon == -1:
            print("Q-Learning Tái tạo (q_study logic): Trạng thái hiện tại không hợp lệ. Dừng.")
            break

        best_action_idx_recon = -1
        max_q_val_recon = -float('inf')
        candidate_actions_recon = []

        for i_rec_act, (dr_rec, dc_rec) in enumerate(moves):
            if 0 <= zero_r_recon + dr_rec < GRID_SIZE and 0 <= zero_c_recon + dc_rec < GRID_SIZE: # If move is valid
                if q_values_recon[i_rec_act] > max_q_val_recon:
                    max_q_val_recon = q_values_recon[i_rec_act]
                    candidate_actions_recon = [i_rec_act]
                elif q_values_recon[i_rec_act] == max_q_val_recon:
                    candidate_actions_recon.append(i_rec_act)

        if candidate_actions_recon:
            best_action_idx_recon = random.choice(candidate_actions_recon) # Handle ties by random choice
        else:
            # This case means no valid moves or all Q-values are -inf (should not happen if Q-table initialized to 0)
            # Fallback: try a random valid move if stuck
            valid_fallback_actions = []
            for i_fb_act, (dr_fb, dc_fb) in enumerate(moves):
                 if 0 <= zero_r_recon + dr_fb < GRID_SIZE and 0 <= zero_c_recon + dc_fb < GRID_SIZE:
                     valid_fallback_actions.append(i_fb_act)
            if valid_fallback_actions:
                best_action_idx_recon = random.choice(valid_fallback_actions)
                # print(f"Q-Learning Tái tạo (q_study logic): Trạng thái {current_s_tuple_recon} có vẻ chưa học rõ. Chọn ngẫu nhiên hợp lệ.")
            else:
                print("Q-Learning Tái tạo (q_study logic): Không có hành động hợp lệ từ trạng thái hiện tại. Dừng.")
                break


        if best_action_idx_recon == -1: # Should be handled by fallback
             print("Q-Learning Tái tạo (q_study logic): Lỗi không xác định được hành động. Dừng.")
             break

        # Perform the chosen action
        dr_best_rec, dc_best_rec = moves[best_action_idx_recon]
        _zr_perform_rec, _zc_perform_rec = find_zero_pos(current_s_list_recon) # zero_r_recon, zero_c_recon

        next_s_list_perform_recon = copy.deepcopy(current_s_list_recon)
        next_s_list_perform_recon[_zr_perform_rec][_zc_perform_rec], \
        next_s_list_perform_recon[_zr_perform_rec + dr_best_rec][_zc_perform_rec + dc_best_rec] = \
            next_s_list_perform_recon[_zr_perform_rec + dr_best_rec][_zc_perform_rec + dc_best_rec], \
            next_s_list_perform_recon[_zr_perform_rec][_zc_perform_rec]

        current_s_list_recon = next_s_list_perform_recon
        current_s_tuple_recon_next = state_to_tuple(current_s_list_recon)

        if current_s_tuple_recon_next in visited_recon_states:
            print("Q-Learning Tái tạo (q_study logic): Phát hiện vòng lặp. Dừng.")
            path_recon.append(copy.deepcopy(current_s_list_recon)) # Add the state that caused the loop
            break
        visited_recon_states.add(current_s_tuple_recon_next)
        path_recon.append(copy.deepcopy(current_s_list_recon))
    else: # Loop finished due to max_reconstruction_steps
        if not path_found_to_goal_ql:
            print(f"Q-Learning Tái tạo (q_study logic): Đạt số bước tối đa ({max_reconstruction_steps}) mà không tìm thấy đích.")

    if q_table_trained_this_session: # Save if Q-table was trained in this session
        # Optionally, only save if a path to the goal was found after training
        # if path_found_to_goal_ql:
        try:
            with open(q_table_filename, 'wb') as f:
                pickle.dump(q_table, f)
            print(f"Đã lưu Q-table (q_study logic) vào file: {q_table_filename}")
        except Exception as e:
            print(f"Lỗi khi lưu Q-table (q_study logic): {e}")
        # else:
        # print(f"Q-table (q_study logic) được training trong phiên này nhưng KHÔNG dẫn đến giải pháp tới đích. KHÔNG lưu Q-table.")

    return path_recon


# --- Hàm Vẽ Pygame ---
def draw_board(screen, state, x_offset, y_offset, font_tile):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            num = state[r][c]
            rect_x = x_offset + c * TILE_SIZE
            rect_y = y_offset + r * TILE_SIZE
            rect = pygame.Rect(rect_x, rect_y, TILE_SIZE, TILE_SIZE)
            if num == 0:
                pygame.draw.rect(screen, EMPTY_TILE_BG_COLOR, rect)
            else:
                pygame.draw.rect(screen, BLUE, rect)
                text_surf = font_tile.render(str(num), True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)
            pygame.draw.rect(screen, TILE_BORDER_COLOR, rect, TILE_BORDER_WIDTH)


def draw_button(screen, text, rect, font_button, normal_color, hover_color, active_color, is_active, is_hover):
    color = normal_color
    shadow_offset = 2
    shadow_color = tuple(max(0, c_val - 40) for c_val in color[:3])
    shadow_rect = rect.move(shadow_offset, shadow_offset)
    pygame.draw.rect(screen, shadow_color, shadow_rect, border_radius=BUTTON_BORDER_RADIUS)

    if is_active:
        color = active_color
    elif is_hover:
        color = hover_color
    pygame.draw.rect(screen, color, rect, border_radius=BUTTON_BORDER_RADIUS)

    text_surf = font_button.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)


def animate_solution(solution_path_anim, screen, font_tile, bg_color=WINDOW_BG_COLOR):
    if not solution_path_anim or len(solution_path_anim) <= 1: return True
    num_steps_anim = len(solution_path_anim) - 1
    delay = NORMAL_ANIMATION_DELAY
    if num_steps_anim > MAX_ANIMATION_STEPS_BEFORE_ACCELERATION:
        delay = FAST_ANIMATION_DELAY
        print(f"Hoạt ảnh: Đường đi dài ({num_steps_anim} bước). Tăng tốc hoạt ảnh.")

    goal_board_area = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)

    for i, state_anim in enumerate(solution_path_anim):
        pygame.draw.rect(screen, bg_color, goal_board_area)
        draw_board(screen, state_anim, BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, font_tile)
        pygame.display.update(goal_board_area)

        quit_attempt_anim = False
        start_wait_anim = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_wait_anim < delay:
            for event_anim_loop in pygame.event.get():
                if event_anim_loop.type == pygame.QUIT:
                    quit_attempt_anim = True
                    break
            if quit_attempt_anim: break
            pygame.time.delay(10)
        if quit_attempt_anim:
            print("Hoạt ảnh bị hủy bởi người dùng.")
            return False
    return True


def format_state_to_string(state):
    if not state: return "[]"
    return " / ".join(" ".join(map(str, row)) for row in state)


def draw_solution_path(screen, font_path, solution_draw, path_area_rect, scroll_y, title_font):
    pygame.draw.rect(screen, PATH_BG_COLOR, path_area_rect)
    pygame.draw.rect(screen, BLACK, path_area_rect, 1)

    path_title_surf = title_font.render("Finish Path", True, TITLE_COLOR)
    path_title_rect = path_title_surf.get_rect(centerx=path_area_rect.centerx,
                                               top=path_area_rect.top - title_font.get_height() - PADDING // 2)
    screen.blit(path_title_surf, path_title_rect)

    if not solution_draw:
        no_sol_text = font_path.render("No solution path to display.", True, PATH_TEXT_COLOR)
        no_sol_rect = no_sol_text.get_rect(center=path_area_rect.center)
        screen.blit(no_sol_text, no_sol_rect)
        return 0

    line_height_path = font_path.get_height()
    line_y_spacing = line_height_path + PATH_LINE_SPACING
    start_draw_x_path = path_area_rect.left + PADDING // 2
    content_top_y_path = path_area_rect.top + PADDING // 2
    total_content_height_path = len(solution_draw) * line_y_spacing - PATH_LINE_SPACING
    first_visible_idx = max(0, int(scroll_y / line_y_spacing))
    lines_to_render_count = int(path_area_rect.height / line_y_spacing) + 2
    last_visible_idx = min(len(solution_draw), first_visible_idx + lines_to_render_count)

    screen.set_clip(path_area_rect)
    for i in range(first_visible_idx, last_visible_idx):
        state_str_path = format_state_to_string(solution_draw[i])
        line_text_path = f"{i}: {state_str_path}"
        draw_y_path = content_top_y_path + (i * line_y_spacing) - scroll_y
        if draw_y_path + line_height_path < path_area_rect.top or draw_y_path > path_area_rect.bottom:
            continue
        text_surf_path = font_path.render(line_text_path, True, PATH_TEXT_COLOR)
        screen.blit(text_surf_path, (start_draw_x_path, draw_y_path))
    screen.set_clip(None)
    return total_content_height_path


def draw_scrollbar(screen, scrollbar_track_rect, scroll_y, total_content_height, visible_area_height, handle_hover):
    if total_content_height <= visible_area_height:
        return None
    pygame.draw.rect(screen, SCROLLBAR_BG_COLOR, scrollbar_track_rect, border_radius=4)
    handle_height_ratio = visible_area_height / total_content_height
    handle_height = max(20, int(scrollbar_track_rect.height * handle_height_ratio))
    handle_height = min(handle_height, scrollbar_track_rect.height)
    scrollable_range = total_content_height - visible_area_height
    track_movement_range = scrollbar_track_rect.height - handle_height
    handle_y_pos = scrollbar_track_rect.top
    if scrollable_range > 0 and track_movement_range > 0:
        scroll_ratio = scroll_y / scrollable_range
        handle_y_pos += int(scroll_ratio * track_movement_range)
    handle_y_pos = max(scrollbar_track_rect.top, min(handle_y_pos, scrollbar_track_rect.bottom - handle_height))
    handle_rect_scroll = pygame.Rect(scrollbar_track_rect.left, handle_y_pos, scrollbar_track_rect.width, handle_height)
    current_handle_color = SCROLLBAR_HANDLE_HOVER_COLOR if handle_hover else SCROLLBAR_HANDLE_COLOR
    pygame.draw.rect(screen, current_handle_color, handle_rect_scroll, border_radius=4)
    return handle_rect_scroll


# --- Hàm Vẽ Biểu đồ ---
def generate_comparison_chart_surface(comparison_data, width, height, chart_title="Algorithm Comparison"):
    if not MATPLOTLIB_AVAILABLE or not comparison_data:
        placeholder_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        placeholder_surf.fill(WINDOW_BG_COLOR)
        font_chart_error = pygame.font.SysFont(FONT_NAME, 16)
        error_text_chart = "Matplotlib không khả dụng." if not MATPLOTLIB_AVAILABLE else "Không có dữ liệu biểu đồ."
        text_surf_chart = font_chart_error.render(error_text_chart, True, RED)
        text_rect_chart = text_surf_chart.get_rect(center=(width // 2, height // 2))
        placeholder_surf.blit(text_surf_chart, text_rect_chart)
        return placeholder_surf

    labels = [res['algo_name'] for res in comparison_data]
    times = [res['time'] if isinstance(res['time'], (int, float)) else 0 for res in comparison_data]
    steps = [res['steps'] if res['steps'] is not None else 0 for res in comparison_data]
    max_time_val = max(times) if times else 1.0
    max_steps_val = max(steps) if steps else 1.0
    if max_time_val == 0: max_time_val = 1.0
    if max_steps_val == 0: max_steps_val = 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100.0, height / 100.0), dpi=100)
    fig_bg_color_mpl = (WINDOW_BG_COLOR[0] / 255, WINDOW_BG_COLOR[1] / 255, WINDOW_BG_COLOR[2] / 255)
    fig.patch.set_facecolor(fig_bg_color_mpl)
    ax1.patch.set_facecolor(fig_bg_color_mpl)
    ax2.patch.set_facecolor(fig_bg_color_mpl)
    bar_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#B19CD9', '#FFD700', '#ADD8E6']

    ax1.bar(labels, times, color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))])
    ax1.set_ylabel('Time (s)', fontsize=9)
    ax1.set_title(f'{chart_title}: Time Taken', fontsize=10)
    ax1.tick_params(axis='x', labelsize=8, rotation=20)
    ax1.tick_params(axis='y', labelsize=8)
    for label_ax1 in ax1.get_xticklabels(): label_ax1.set_horizontalalignment('right')
    for i, v_time in enumerate(times):
        time_display_text = f"{v_time:.2f}s"
        error_msg_check_time = comparison_data[i].get('error')
        if error_msg_check_time and "Timed out" in error_msg_check_time:
            time_display_text = f"TIMEOUT\n({v_time:.0f}s)"
        ax1.text(i, v_time + 0.02 * max_time_val, time_display_text, color='black', ha='center', va='bottom',
                 fontsize=7)

    ax2.bar(labels, steps, color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))])
    ax2.set_ylabel('Steps (g_cost / path length)', fontsize=9)
    ax2.set_title(f'{chart_title}: Steps in Path', fontsize=10)
    ax2.tick_params(axis='x', labelsize=8, rotation=20)
    ax2.tick_params(axis='y', labelsize=8)
    for label_ax2 in ax2.get_xticklabels(): label_ax2.set_horizontalalignment('right')
    for i, v_steps in enumerate(steps):
        step_display_text = str(v_steps)
        error_msg_check_steps = comparison_data[i].get('error')
        goal_reached_check = comparison_data[i].get('goal_reached', False)
        final_h_check = comparison_data[i].get('final_h')
        if error_msg_check_steps:
            if "Timed out" in error_msg_check_steps:
                step_display_text = "N/A (Timeout)"
            elif ("No solution" in error_msg_check_steps or "Path ended" in error_msg_check_steps or
                  "error during run" in error_msg_check_steps or "Lỗi Bộ Nhớ" in error_msg_check_steps):
                step_display_text = "N/A"
        elif not goal_reached_check and final_h_check is not None:
            step_display_text = f"{v_steps} (H:{final_h_check})"
        ax2.text(i, v_steps + 0.02 * max_steps_val, step_display_text, color='black', ha='center', va='bottom',
                 fontsize=7)

    plt.tight_layout(pad=1.0, h_pad=2.0)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_surface = pygame.image.load(buf, 'png').convert_alpha()
    except Exception as e_chart:
        print(f"Lỗi khi tạo biểu đồ matplotlib: {e_chart}")
        chart_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        chart_surface.fill(WINDOW_BG_COLOR)
        font_chart_error_disp = pygame.font.SysFont(FONT_NAME, 14)
        text_surf_err_disp = font_chart_error_disp.render(f"Lỗi tạo biểu đồ: {str(e_chart)[:50]}", True, RED)
        chart_surface.blit(text_surf_err_disp, text_surf_err_disp.get_rect(center=(width // 2, height // 2)))
    finally:
        plt.close(fig)
        buf.close()
    return chart_surface


def display_results(screen, font_info, algo_name, time_taken, steps, message=None, start_y=500, start_x=50,
                    bg_color=WINDOW_BG_COLOR, final_g_cost=None, final_h_cost=None,
                    comparison_chart_surface=None, chart_display_rect=None):
    clear_width = (BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH) - BOARD_OFFSET_X_START
    clear_height = CHART_HEIGHT + PADDING
    clear_rect_results = pygame.Rect(start_x - PADDING // 2, start_y - PADDING // 2, clear_width, clear_height)
    pygame.draw.rect(screen, bg_color, clear_rect_results)

    if comparison_chart_surface and chart_display_rect:
        screen.blit(comparison_chart_surface, chart_display_rect.topleft)
    elif message:
        current_y_msg = start_y
        line_height_msg = font_info.get_height() + 3
        max_lines_msg = 7
        for i_msg, line_content in enumerate(message.split('\n')[:max_lines_msg]):
            display_line_msg = (line_content[:90] + '...') if len(line_content) > 93 else line_content
            text_surf_msg = font_info.render(display_line_msg, True,
                                             RED if "Lỗi" in message or "Error" in message else INFO_TEXT_COLOR)
            screen.blit(text_surf_msg, (start_x, current_y_msg))
            current_y_msg += line_height_msg
        if len(message.split('\n')) > max_lines_msg:
            screen.blit(font_info.render("...", True, INFO_TEXT_COLOR), (start_x, current_y_msg))
    elif algo_name:
        current_y_single = start_y
        line_height_single = font_info.get_height() + 3
        infos_to_display = [
            f"Algorithm: {algo_name}",
            f"Time: {time_taken:.4f} seconds",
            f"Steps (path length / g_cost): {steps if steps is not None else 'N/A'}"
        ]
        if final_h_cost is not None: infos_to_display.append(f"Final State Heuristic (h): {final_h_cost}")
        if algo_name in ["A_Star", "IDA"] and final_g_cost is not None and final_h_cost is not None:
            infos_to_display.append(f"Final f_cost (g+h): {final_g_cost + final_h_cost}")
        for line_info in infos_to_display:
            text_surf_info = font_info.render(line_info, True, INFO_TEXT_COLOR)
            screen.blit(text_surf_info, (start_x, current_y_single))
            current_y_single += line_height_single


# --- Hàm lưu kết quả test ---
def save_test_results_to_file(filename, timestamp, algo_name, initial_s, goal_s,
                              time_t, steps_g, sol_path, err_msg, final_h_val, final_g_val,
                              target_s_for_status):
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write("----------------------------------------\n")
            f.write(f"Test Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Algorithm: {algo_name}\n")
            f.write(f"Initial State: {format_state_to_string(initial_s)}\n")
            f.write(f"Goal State: {format_state_to_string(goal_s)}\n")
            if err_msg and "Timed out" in err_msg:
                f.write(f"Time Taken: {err_msg}\n")
            else:
                f.write(f"Time Taken: {time_t:.4f} seconds\n")
            f.write(f"Steps in Path (g_cost / length): {steps_g if steps_g is not None else 'N/A'}\n")
            goal_reached_log = "N/A"
            if sol_path and sol_path[-1] == target_s_for_status:
                goal_reached_log = "Yes"
            elif sol_path:
                goal_reached_log = f"Partial (Path ends, H={manhattan_distance(sol_path[-1], target_s_for_status)})"
            elif err_msg and "Timed out" in err_msg:
                goal_reached_log = "No (Timed out)"
            elif err_msg:
                goal_reached_log = f"No ({err_msg.splitlines()[0][:50]})"
            else:
                goal_reached_log = "No solution path found"
            f.write(f"Goal Reached: {goal_reached_log}\n")
            f.write(
                f"Final State Heuristic (h of last state in path): {final_h_val if final_h_val is not None else 'N/A'}\n")
            if algo_name in ["A_Star", "IDA"] and final_g_val is not None and final_h_val is not None:
                f.write(f"Final State f_cost (g+h): {final_g_val + final_h_val}\n")
            f.write(f"Error Message Logged: {err_msg if err_msg else 'None'}\n")
            if sol_path:
                f.write("Solution Path (actual path taken/found):\n")
                if len(sol_path) > 200 or (err_msg and "Yes" not in goal_reached_log):
                    f.write(
                        f"(Path may be long or incomplete: {len(sol_path) - 1 if sol_path else 0} steps. Summary shown.)\n")
                    if sol_path:
                        for i, state_item in enumerate(sol_path[:5]): f.write(
                            f"{i}: {format_state_to_string(state_item)}\n")
                        if len(sol_path) > 10: f.write("...\n")
                        for i, state_item in enumerate(sol_path[-5:], start=max(5, len(sol_path) - 5)): f.write(
                            f"{i}: {format_state_to_string(state_item)}\n")
                else:
                    for i, state_item in enumerate(sol_path): f.write(f"{i}: {format_state_to_string(state_item)}\n")
            else:
                f.write("Solution Path: None\n")
            f.write("----------------------------------------\n\n")
        print(f"Kết quả đã được lưu vào file: {filename}")
    except Exception as e_log:
        print(f"Lỗi khi lưu kết quả vào file: {e_log}")


# --- Vòng lặp Chính của Trò chơi ---
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Trình Giải 8-Puzzle - Nhóm 3 Local Search Added")
    clock = pygame.time.Clock()

    try:
        font_tile = pygame.font.SysFont(FONT_NAME, FONT_SIZE_TILE, bold=True)
        font_button = pygame.font.SysFont(FONT_NAME, FONT_SIZE_BUTTON)
        font_info = pygame.font.SysFont(FONT_NAME, FONT_SIZE_INFO)
        font_title = pygame.font.SysFont(FONT_NAME, FONT_SIZE_TITLE, bold=True)
        font_path = pygame.font.SysFont(FONT_NAME, FONT_SIZE_PATH)
    except Exception as e_font:
        print(f"Lỗi tải font hệ thống '{FONT_NAME}': {e_font}. Sử dụng font mặc định của Pygame.")
        font_tile = pygame.font.Font(None, FONT_SIZE_TILE + 10)
        font_button = pygame.font.Font(None, FONT_SIZE_BUTTON + 4)
        font_info = pygame.font.Font(None, FONT_SIZE_INFO + 6)
        font_title = pygame.font.Font(None, FONT_SIZE_TITLE + 8)
        font_path = pygame.font.Font(None, FONT_SIZE_PATH + 4)

    current_initial_state = copy.deepcopy(begin_state)
    current_goal_state_ref = copy.deepcopy(goal_state)
    solution, elapsed_time, num_steps = None, 0, 0
    active_button_algo, last_displayed_algo_type, error_message = None, None, None
    is_animating, solving_in_progress = False, False
    final_g, final_h = None, None
    comparison_results_data, current_comparison_chart_surface = None, None
    path_scroll_y, path_total_content_height = 0, 0
    scrollbar_track_rect = pygame.Rect(SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT)
    scrollbar_handle_rect, dragging_scrollbar, scrollbar_mouse_offset_y = None, False, 0

    button_texts_main = [
        "BFS", "UCS", "DFS", "IDDFS", "A_Star", "IDA", "Greedy",
        "Simple HC", "Steepest HC", "Stochastic HC", "SA",
        "Beam Search", "Genetic Algo", "Q-Learning",
        "Compare Informed", "Compare Uninformed", "Compare Local Search",
        "Open Group 4", "Open Group 5"
    ]
    button_rects_main = {}
    num_buttons_main = len(button_texts_main)
    cols_main = 2
    base_rows_per_col_main = num_buttons_main // cols_main
    extra_buttons_main = num_buttons_main % cols_main
    col1_count_main = base_rows_per_col_main + (1 if extra_buttons_main > 0 else 0)
    current_col_x_main, row_in_col_main, col_num_main = BUTTON_START_X_COL1, 0, 1
    for i, text_btn_main in enumerate(button_texts_main):
        if col_num_main == 1 and row_in_col_main >= col1_count_main:
            current_col_x_main, row_in_col_main, col_num_main = BUTTON_START_X_COL2, 0, 2
        y_btn = BUTTON_COL_START_Y + row_in_col_main * (BUTTON_HEIGHT + BUTTON_PADDING)
        button_rects_main[text_btn_main] = pygame.Rect(current_col_x_main, y_btn, BUTTON_WIDTH, BUTTON_HEIGHT)
        row_in_col_main += 1

    all_button_bottom_y = max(
        rect.bottom for rect in button_rects_main.values()) if button_rects_main else BUTTON_COL_START_Y
    run_button_y_pos = all_button_bottom_y + PADDING * 2
    run_rect = pygame.Rect(BUTTON_START_X_COL1, run_button_y_pos, BUTTON_WIDTH, BUTTON_HEIGHT)
    reset_rect = pygame.Rect(BUTTON_START_X_COL2, run_button_y_pos, BUTTON_WIDTH, BUTTON_HEIGHT)
    controls_area_width = (BUTTON_START_X_COL2 + BUTTON_WIDTH) - BUTTON_START_X_COL1
    controls_title_center_x = BUTTON_START_X_COL1 + controls_area_width // 2
    controls_title_y_bottom = BUTTON_COL_START_Y - PADDING
    results_area_x, results_area_y = BOARD_OFFSET_X_START, BOARD_OFFSET_Y + BOARD_TOTAL_HEIGHT + PADDING * 4
    results_title_y = results_area_y - font_title.get_height() - PADDING // 2
    chart_display_rect = pygame.Rect(results_area_x, results_area_y, CHART_WIDTH, CHART_HEIGHT)
    path_area_rect = pygame.Rect(PATH_AREA_X_START, PATH_AREA_Y_START, PATH_AREA_WIDTH, PATH_AREA_HEIGHT)
    start_board_rect = pygame.Rect(BOARD_OFFSET_X_START, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)
    goal_board_rect = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)

    algorithm_map_main = {
        "BFS": BFS, "UCS": UCS, "DFS": DFS, "IDDFS": IDDFS,
        "A_Star": A_Star, "IDA": IDA, "Greedy": Greedy,
        "Simple HC": SimpleHillClimbing, "Steepest HC": SteepestHillClimbing,
        "Stochastic HC": StochasticHillClimbing, "SA": SimulatedAnnealing,
        "Beam Search": lambda start_bs, goal_bs: BeamSearch(start_bs, goal_bs, BEAM_WIDTH),
        "Genetic Algo": GeneticAlgorithm, "Q-Learning": QLearning
    }
    informed_group_to_compare = ["Greedy", "A_Star", "IDA"]
    uninformed_group_to_compare = ["BFS", "UCS", "DFS", "IDDFS"]
    local_search_group_to_compare = ["Simple HC", "Steepest HC", "Stochastic HC", "SA", "Beam Search", "Genetic Algo"]

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered_button_main, scrollbar_handle_hover = None, False
        events_list = pygame.event.get()
        for event_loop in events_list:
            if event_loop.type == pygame.QUIT: running = False; break
        if not running: break

        if not is_animating and not solving_in_progress:
            for event_handler in events_list:
                if event_handler.type == pygame.MOUSEBUTTONDOWN and event_handler.button == 1:
                    clicked_this_loop = False
                    if scrollbar_handle_rect and scrollbar_handle_rect.collidepoint(event_handler.pos):
                        dragging_scrollbar, scrollbar_mouse_offset_y = True, event_handler.pos[
                                                                                 1] - scrollbar_handle_rect.top
                        clicked_this_loop = True
                    elif scrollbar_track_rect.collidepoint(
                            event_handler.pos) and path_total_content_height > PATH_AREA_HEIGHT:
                        if scrollbar_handle_rect:
                            relative_y_track_click = (event_handler.pos[
                                                          1] - scrollbar_track_rect.top) / scrollbar_track_rect.height
                            scrollable_range_track = path_total_content_height - PATH_AREA_HEIGHT
                            target_scroll_y_track = (relative_y_track_click * scrollable_range_track) - (
                                        PATH_AREA_HEIGHT / 2)
                            path_scroll_y = max(0, min(target_scroll_y_track,
                                                       scrollable_range_track if scrollable_range_track > 0 else 0))
                        clicked_this_loop = True
                    if not clicked_this_loop:
                        for text_btn_click, rect_btn_click in button_rects_main.items():
                            if rect_btn_click.collidepoint(event_handler.pos):
                                if text_btn_click == "Open Group 4" or text_btn_click == "Open Group 5":
                                    running = False
                                    pygame.quit()
                                    script_name = "group4_algo.py" if text_btn_click == "Open Group 4" else "group5_algo.py"
                                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
                                    os.system(f"python \"{script_path}\"")  # Added quotes for paths with spaces
                                    # This process will exit. The other script runs as a new process.
                                    sys.exit()  # Ensure current script fully exits
                                if active_button_algo != text_btn_click:
                                    active_button_algo = text_btn_click
                                    print(f"Đã chọn thuật toán/chức năng: {active_button_algo}")
                                clicked_this_loop = True
                                break
                    if not clicked_this_loop and run_rect.collidepoint(event_handler.pos):
                        if active_button_algo:
                            print(f"Đang chạy {active_button_algo}...")
                            solution, error_message, comparison_results_data, current_comparison_chart_surface = None, None, None, None
                            elapsed_time, num_steps, path_scroll_y, final_g, final_h = 0, 0, 0, None, None
                            solving_in_progress = True
                            results_clear_rect = chart_display_rect.inflate(PADDING, PADDING)
                            path_clear_area = path_area_rect.inflate(0, font_title.get_height() + PADDING).union(
                                scrollbar_track_rect)
                            screen.fill(WINDOW_BG_COLOR, results_clear_rect)
                            screen.fill(WINDOW_BG_COLOR, path_clear_area)
                            solving_msg_surf = font_info.render("Solving...", True, INFO_TEXT_COLOR)
                            screen.blit(solving_msg_surf, solving_msg_surf.get_rect(center=chart_display_rect.center))
                            pygame.display.update([results_clear_rect, path_clear_area, goal_board_rect])
                        else:
                            error_message, last_displayed_algo_type = "Vui lòng chọn một thuật toán hoặc chức năng trước.", "Error"
                        clicked_this_loop = True
                    if not clicked_this_loop and reset_rect.collidepoint(event_handler.pos):
                        current_initial_state, current_goal_state_ref = copy.deepcopy(begin_state), copy.deepcopy(
                            goal_state)
                        solution, active_button_algo, last_displayed_algo_type, error_message = None, None, None, None
                        comparison_results_data, current_comparison_chart_surface = None, None
                        elapsed_time, num_steps, path_scroll_y, final_g, final_h = 0, 0, 0, None, None
                        is_animating, solving_in_progress = False, False
                        print("Trò chơi đã được reset.")
                        clicked_this_loop = True
                    if not clicked_this_loop and start_board_rect.collidepoint(event_handler.pos):
                        col_s, row_s = (event_handler.pos[0] - BOARD_OFFSET_X_START) // TILE_SIZE, (
                                    event_handler.pos[1] - BOARD_OFFSET_Y) // TILE_SIZE
                        if 0 <= row_s < GRID_SIZE and 0 <= col_s < GRID_SIZE:
                            zr_s, zc_s = find_zero_pos(current_initial_state)
                            if zr_s != -1 and abs(row_s - zr_s) + abs(col_s - zc_s) == 1:
                                current_initial_state[zr_s][zc_s], current_initial_state[row_s][col_s] = \
                                current_initial_state[row_s][col_s], current_initial_state[zr_s][zc_s]
                                solution, last_displayed_algo_type, error_message, comparison_results_data, current_comparison_chart_surface = None, None, None, None, None
                                path_scroll_y, final_g, final_h = 0, None, None
                        clicked_this_loop = True
                    if not clicked_this_loop and goal_board_rect.collidepoint(event_handler.pos):
                        col_g, row_g = (event_handler.pos[0] - BOARD_OFFSET_X_GOAL) // TILE_SIZE, (
                                    event_handler.pos[1] - BOARD_OFFSET_Y) // TILE_SIZE
                        if 0 <= row_g < GRID_SIZE and 0 <= col_g < GRID_SIZE:
                            zr_g, zc_g = find_zero_pos(current_goal_state_ref)
                            if zr_g != -1 and abs(row_g - zr_g) + abs(col_g - zc_g) == 1:
                                current_goal_state_ref[zr_g][zc_g], current_goal_state_ref[row_g][col_g] = \
                                current_goal_state_ref[row_g][col_g], current_goal_state_ref[zr_g][zc_g]
                                print(f"Trạng thái đích được sửa đổi: {format_state_to_string(current_goal_state_ref)}")
                                solution, last_displayed_algo_type, error_message, comparison_results_data, current_comparison_chart_surface = None, None, None, None, None
                                path_scroll_y, final_g, final_h = 0, None, None
                        clicked_this_loop = True
                elif event_handler.type == pygame.MOUSEBUTTONUP and event_handler.button == 1:
                    if dragging_scrollbar: dragging_scrollbar = False
                elif event_handler.type == pygame.MOUSEMOTION:
                    if dragging_scrollbar and scrollbar_handle_rect:
                        new_handle_top_drag = event_handler.pos[1] - scrollbar_mouse_offset_y
                        new_handle_top_drag = max(scrollbar_track_rect.top, min(new_handle_top_drag,
                                                                                scrollbar_track_rect.bottom - scrollbar_handle_rect.height))
                        scrollable_range_drag, track_movement_drag = path_total_content_height - PATH_AREA_HEIGHT, scrollbar_track_rect.height - scrollbar_handle_rect.height
                        if track_movement_drag > 0 and scrollable_range_drag > 0:
                            relative_handle_pos_drag = (
                                                                   new_handle_top_drag - scrollbar_track_rect.top) / track_movement_drag
                            path_scroll_y = max(0, min(relative_handle_pos_drag * scrollable_range_drag,
                                                       scrollable_range_drag))

        if solving_in_progress:
            current_comparison_chart_surface, comparison_results_data, error_message = None, None, None
            solution, path_scroll_y, final_g, final_h = None, 0, None, None
            if active_button_algo == "Compare Uninformed":
                last_displayed_algo_type, temp_comparison_results = "Compare Uninformed", []
                print("Đang chạy So sánh Thuật toán Không Thông tin...")
                for algo_name_comp_ui in uninformed_group_to_compare:
                    print(f"So sánh (Uninformed): {algo_name_comp_ui} (Timeout: {MAX_UNINFORMED_ALGO_RUNTIME}s)")
                    solver_func_comp_ui, sol_comp_ui, err_comp_ui, time_comp_ui = algorithm_map_main.get(
                        algo_name_comp_ui), None, None, MAX_UNINFORMED_ALGO_RUNTIME
                    steps_comp_ui, f_g_comp_ui, f_h_comp_ui, goal_reached_ui = 0, None, None, False
                    run_result_obj_ui = AlgorithmRunResult()
                    if solver_func_comp_ui:
                        thread_ui = threading.Thread(target=run_algorithm_in_thread_wrapper,
                                                     args=(solver_func_comp_ui, current_initial_state,
                                                           current_goal_state_ref, run_result_obj_ui))
                        start_time_thread_ui = time.time()
                        thread_ui.start()
                        thread_ui.join(MAX_UNINFORMED_ALGO_RUNTIME)
                        time_comp_ui = time.time() - start_time_thread_ui
                        if thread_ui.is_alive():
                            err_comp_ui = f"Timed out after {MAX_UNINFORMED_ALGO_RUNTIME}s"
                        else:
                            sol_comp_ui, err_comp_ui = run_result_obj_ui.solution, run_result_obj_ui.error_message if run_result_obj_ui.exception_occurred else None
                    else:
                        err_comp_ui, time_comp_ui = "Lỗi: Hàm không tìm thấy", 0
                    if sol_comp_ui:
                        steps_comp_ui, f_g_comp_ui = (len(sol_comp_ui) - 1 if sol_comp_ui else 0), (
                            len(sol_comp_ui) - 1 if sol_comp_ui else 0)
                        f_h_comp_ui, goal_reached_ui = manhattan_distance(sol_comp_ui[-1], current_goal_state_ref), (
                                    sol_comp_ui[-1] == current_goal_state_ref)
                        if not err_comp_ui and not goal_reached_ui: err_comp_ui = f"Path ends H={f_h_comp_ui}"
                    elif not err_comp_ui:
                        err_comp_ui = "No solution path found."
                    if not sol_comp_ui: f_h_comp_ui = manhattan_distance(current_initial_state, current_goal_state_ref)
                    save_test_results_to_file(LOG_FILE_NAME, datetime.datetime.now(), algo_name_comp_ui,
                                              current_initial_state, current_goal_state_ref, time_comp_ui, f_g_comp_ui,
                                              sol_comp_ui, err_comp_ui, f_h_comp_ui, f_g_comp_ui,
                                              current_goal_state_ref)
                    temp_comparison_results.append(
                        {'algo_name': algo_name_comp_ui, 'time': time_comp_ui, 'steps': steps_comp_ui,
                         'final_h': f_h_comp_ui, 'goal_reached': goal_reached_ui, 'error': err_comp_ui})
                comparison_results_data = temp_comparison_results
                if MATPLOTLIB_AVAILABLE:
                    current_comparison_chart_surface = generate_comparison_chart_surface(comparison_results_data,
                                                                                         CHART_WIDTH, CHART_HEIGHT,
                                                                                         "Uninformed Search Comparison")
                elif not error_message:
                    error_message = "Matplotlib không khả dụng để vẽ biểu đồ."
            elif active_button_algo == "Compare Informed":
                last_displayed_algo_type, temp_comparison_results = "Compare Informed", []
                print("Đang chạy So sánh Thuật toán Thông tin...")
                for algo_name_comp_i in informed_group_to_compare:
                    print(f"So sánh (Informed): {algo_name_comp_i}")
                    solver_func_comp_i, sol_comp_i, err_comp_i = algorithm_map_main.get(algo_name_comp_i), None, None
                    steps_comp_i, f_g_comp_i, f_h_comp_i, goal_reached_i = 0, None, None, False
                    start_time_comp_i = time.time()
                    try:
                        if solver_func_comp_i:
                            sol_comp_i = solver_func_comp_i(current_initial_state, current_goal_state_ref)
                        else:
                            err_comp_i = "Lỗi: Hàm không tìm thấy"
                    except Exception as e_comp_i:
                        err_comp_i = f"Runtime Error: {type(e_comp_i).__name__}: {str(e_comp_i)[:100]}"; traceback.print_exc()
                    time_comp_i = time.time() - start_time_comp_i
                    if sol_comp_i:
                        steps_comp_i, f_g_comp_i = (len(sol_comp_i) - 1 if sol_comp_i else 0), (
                            len(sol_comp_i) - 1 if sol_comp_i else 0)
                        f_h_comp_i, goal_reached_i = manhattan_distance(sol_comp_i[-1], current_goal_state_ref), (
                                    sol_comp_i[-1] == current_goal_state_ref)
                        if not err_comp_i and not goal_reached_i: err_comp_i = f"Path ends H={f_h_comp_i}"
                    elif not err_comp_i:
                        err_comp_i = "No solution path found."
                    if not sol_comp_i: f_h_comp_i = manhattan_distance(current_initial_state, current_goal_state_ref)
                    save_test_results_to_file(LOG_FILE_NAME, datetime.datetime.now(), algo_name_comp_i,
                                              current_initial_state, current_goal_state_ref, time_comp_i, f_g_comp_i,
                                              sol_comp_i, err_comp_i, f_h_comp_i, f_g_comp_i, current_goal_state_ref)
                    temp_comparison_results.append(
                        {'algo_name': algo_name_comp_i, 'time': time_comp_i, 'steps': steps_comp_i,
                         'final_h': f_h_comp_i, 'goal_reached': goal_reached_i, 'error': err_comp_i})
                comparison_results_data = temp_comparison_results
                if MATPLOTLIB_AVAILABLE:
                    current_comparison_chart_surface = generate_comparison_chart_surface(comparison_results_data,
                                                                                         CHART_WIDTH, CHART_HEIGHT,
                                                                                         "Informed Search Comparison")
                elif not error_message:
                    error_message = "Matplotlib không khả dụng để vẽ biểu đồ."
            elif active_button_algo == "Compare Local Search":
                last_displayed_algo_type, temp_comparison_results = "Compare Local Search", []
                print("Đang chạy So sánh Thuật toán Local Search...")
                for algo_name_comp_ls in local_search_group_to_compare:
                    print(f"So sánh (Local Search): {algo_name_comp_ls} (Timeout: {MAX_LOCAL_SEARCH_ALGO_RUNTIME}s)")
                    solver_func_comp_ls, sol_comp_ls, err_comp_ls, time_comp_ls = algorithm_map_main.get(
                        algo_name_comp_ls), None, None, MAX_LOCAL_SEARCH_ALGO_RUNTIME
                    steps_comp_ls, f_g_comp_ls, f_h_comp_ls, goal_reached_ls = 0, None, None, False
                    run_result_obj_ls = AlgorithmRunResult()
                    if solver_func_comp_ls:
                        thread_ls = threading.Thread(target=run_algorithm_in_thread_wrapper,
                                                     args=(solver_func_comp_ls, current_initial_state,
                                                           current_goal_state_ref, run_result_obj_ls))
                        start_time_thread_ls = time.time()
                        thread_ls.start()
                        thread_ls.join(MAX_LOCAL_SEARCH_ALGO_RUNTIME)
                        time_comp_ls = time.time() - start_time_thread_ls
                        if thread_ls.is_alive():
                            err_comp_ls = f"Timed out after {MAX_LOCAL_SEARCH_ALGO_RUNTIME}s"
                        else:
                            sol_comp_ls, err_comp_ls = run_result_obj_ls.solution, run_result_obj_ls.error_message if run_result_obj_ls.exception_occurred else None
                    else:
                        err_comp_ls, time_comp_ls = "Lỗi: Hàm không tìm thấy", 0
                    if sol_comp_ls:
                        steps_comp_ls, f_g_comp_ls = (len(sol_comp_ls) - 1 if sol_comp_ls else 0), (
                            len(sol_comp_ls) - 1 if sol_comp_ls else 0)
                        f_h_comp_ls, goal_reached_ls = manhattan_distance(sol_comp_ls[-1], current_goal_state_ref), (
                                    sol_comp_ls[-1] == current_goal_state_ref)
                        if not err_comp_ls and not goal_reached_ls: err_comp_ls = f"Path ends H={f_h_comp_ls}"
                    elif not err_comp_ls:
                        err_comp_ls = "No solution path returned or error."
                    if not sol_comp_ls: f_h_comp_ls = manhattan_distance(current_initial_state, current_goal_state_ref)
                    save_test_results_to_file(LOG_FILE_NAME, datetime.datetime.now(), algo_name_comp_ls,
                                              current_initial_state, current_goal_state_ref, time_comp_ls, f_g_comp_ls,
                                              sol_comp_ls, err_comp_ls, f_h_comp_ls, f_g_comp_ls,
                                              current_goal_state_ref)
                    temp_comparison_results.append(
                        {'algo_name': algo_name_comp_ls, 'time': time_comp_ls, 'steps': steps_comp_ls,
                         'final_h': f_h_comp_ls, 'goal_reached': goal_reached_ls, 'error': err_comp_ls})
                comparison_results_data = temp_comparison_results
                if MATPLOTLIB_AVAILABLE:
                    current_comparison_chart_surface = generate_comparison_chart_surface(comparison_results_data,
                                                                                         CHART_WIDTH, CHART_HEIGHT,
                                                                                         "Local Search Algorithm Comparison")
                elif not error_message:
                    error_message = "Matplotlib không khả dụng để vẽ biểu đồ."
            else:  # Single algorithm run
                last_displayed_algo_type = "Single"
                start_time_single = time.time()
                solver_func_single, temp_sol_single, temp_err_single = algorithm_map_main.get(
                    active_button_algo), None, None
                try:
                    if solver_func_single:
                        temp_sol_single = solver_func_single(current_initial_state, current_goal_state_ref)
                    else:
                        temp_err_single = f"Lỗi: Hàm cho '{active_button_algo}' không tìm thấy!"
                except Exception as e_single:
                    temp_err_single = f"Lỗi Runtime ({active_button_algo}):\n{type(e_single).__name__}: {str(e_single)[:150]}"; traceback.print_exc()
                elapsed_time, solution, error_message = time.time() - start_time_single, temp_sol_single, temp_err_single
                if solution:
                    num_steps, final_g = (len(solution) - 1 if solution else 0), (len(solution) - 1 if solution else 0)
                    final_h = manhattan_distance(solution[-1], current_goal_state_ref)
                    if solution[-1] == current_goal_state_ref:
                        error_message = None
                        if len(solution) > 1: is_animating = True
                    elif not error_message:
                        error_message = f"{active_button_algo}: Path (len {num_steps}), ends H={final_h} (not goal)."
                elif not error_message:
                    error_message = f"{active_button_algo}: Không tìm thấy đường đi giải pháp."
                if not solution: final_h = manhattan_distance(current_initial_state, current_goal_state_ref)
                save_test_results_to_file(LOG_FILE_NAME, datetime.datetime.now(), active_button_algo,
                                          current_initial_state, current_goal_state_ref, elapsed_time, final_g,
                                          solution, error_message, final_h, final_g, current_goal_state_ref)
            solving_in_progress = False

        if not is_animating and not solving_in_progress:
            if scrollbar_handle_rect and scrollbar_handle_rect.collidepoint(mouse_pos):
                scrollbar_handle_hover = True
            else:
                for text_h, rect_h in button_rects_main.items():
                    if rect_h.collidepoint(mouse_pos): hovered_button_main = text_h; break
                if not hovered_button_main:
                    if run_rect.collidepoint(mouse_pos):
                        hovered_button_main = "Run"
                    elif reset_rect.collidepoint(mouse_pos):
                        hovered_button_main = "Reset"
        screen.fill(WINDOW_BG_COLOR)
        screen.blit(font_title.render("Begin State", True, TITLE_COLOR), (BOARD_OFFSET_X_START, PADDING * 2))
        screen.blit(font_title.render("Goal State", True, TITLE_COLOR), (BOARD_OFFSET_X_GOAL, PADDING * 2))
        controls_title_surf = font_title.render("Controller", True, TITLE_COLOR)
        screen.blit(controls_title_surf,
                    controls_title_surf.get_rect(centerx=controls_title_center_x, bottom=controls_title_y_bottom))
        screen.blit(font_title.render("Result", True, TITLE_COLOR), (results_area_x, results_title_y))
        draw_board(screen, current_initial_state, BOARD_OFFSET_X_START, BOARD_OFFSET_Y, font_tile)
        display_state_on_goal_board = current_goal_state_ref
        if last_displayed_algo_type == "Single" and solution and not is_animating: display_state_on_goal_board = \
        solution[-1]
        if not is_animating: draw_board(screen, display_state_on_goal_board, BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y,
                                        font_tile)
        for text_draw_btn, rect_draw_btn in button_rects_main.items():
            is_active_btn, is_hover_btn = (active_button_algo == text_draw_btn), (hovered_button_main == text_draw_btn)
            btn_normal_color = STEEL_BLUE
            if text_draw_btn in ["Compare Informed", "Compare Uninformed",
                                 "Compare Local Search"]: btn_normal_color = MEDIUM_SEA_GREEN
            draw_button(screen, text_draw_btn, rect_draw_btn, font_button,
                        btn_normal_color if not is_active_btn else ORANGE, LIGHT_BLUE, ORANGE, is_active_btn,
                        is_hover_btn)
        draw_button(screen, "Run", run_rect, font_button, GREEN, LIGHT_BLUE, ORANGE, False,
                    hovered_button_main == "Run")
        draw_button(screen, "Reset", reset_rect, font_button, RED, LIGHT_BLUE, ORANGE, False,
                    hovered_button_main == "Reset")

        if last_displayed_algo_type in ["Compare Informed", "Compare Uninformed", "Compare Local Search"]:
            display_results(screen, font_info, None, 0, 0,
                            message=error_message if not current_comparison_chart_surface and error_message else None,
                            start_y=results_area_y, start_x=results_area_x,
                            comparison_chart_surface=current_comparison_chart_surface,
                            chart_display_rect=chart_display_rect if current_comparison_chart_surface else None)
        elif last_displayed_algo_type == "Single":
            display_results(screen, font_info, active_button_algo, elapsed_time, num_steps, message=error_message,
                            start_y=results_area_y, start_x=results_area_x, final_g_cost=final_g, final_h_cost=final_h)
        elif last_displayed_algo_type == "Error":
            display_results(screen, font_info, None, 0, 0, message=error_message, start_y=results_area_y,
                            start_x=results_area_x)

        if last_displayed_algo_type == "Single" and solution:
            path_total_content_height = draw_solution_path(screen, font_path, solution, path_area_rect, path_scroll_y,
                                                           font_title)
            if path_total_content_height > PATH_AREA_HEIGHT:
                scrollbar_handle_rect = draw_scrollbar(screen, scrollbar_track_rect, path_scroll_y,
                                                       path_total_content_height, PATH_AREA_HEIGHT,
                                                       scrollbar_handle_hover or dragging_scrollbar)
            else:
                scrollbar_handle_rect = None
        else:
            pygame.draw.rect(screen, PATH_BG_COLOR, path_area_rect)
            pygame.draw.rect(screen, BLACK, path_area_rect, 1)
            path_title_surf_empty = font_title.render("Finish Path", True, TITLE_COLOR)
            path_title_rect_empty = path_title_surf_empty.get_rect(centerx=path_area_rect.centerx,
                                                                   top=path_area_rect.top - font_title.get_height() - PADDING // 2)
            screen.blit(path_title_surf_empty, path_title_rect_empty)
            if last_displayed_algo_type and "Compare" in last_displayed_algo_type:
                comp_msg_surf = font_path.render("Path details in log for comparisons.", True, PATH_TEXT_COLOR)
                screen.blit(comp_msg_surf, comp_msg_surf.get_rect(center=path_area_rect.center))
            path_total_content_height, scrollbar_handle_rect = 0, None

        if is_animating:
            animation_ok = False
            if solution and solution[-1] == current_goal_state_ref:
                animation_ok = animate_solution(solution, screen, font_tile, bg_color=WINDOW_BG_COLOR)
                if not animation_ok: running = False
            is_animating = False
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    print("Pygame đã đóng thành công.")


if __name__ == "__main__":
    main()
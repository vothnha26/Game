# -*- coding: utf-8 -*-
import pygame
import heapq
from collections import deque
import copy
from queue import PriorityQueue
import time
import random # Cần cho Stochastic HC, SA, GA
import math   # Cần cho Simulated Annealing (SA)
import traceback # Để in lỗi chi tiết

# --- Hằng số ---
# Màu sắc
WINDOW_BG_COLOR = (240, 240, 240) # Màu nền cửa sổ
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 128, 255) # Màu ô số
TILE_BORDER_COLOR = BLACK # Màu viền ô
EMPTY_TILE_BG_COLOR = (200, 200, 200) # Màu ô trống
LIGHT_BLUE = (100, 149, 237) # Màu khi di chuột qua nút
STEEL_BLUE = (80, 140, 190)  # Màu nút bình thường
ORANGE = (255, 140, 0)      # Màu nút đang hoạt động/được chọn
GREEN = (0, 150, 0)        # Màu nút Run
RED = (200, 0, 0)           # Màu nút Reset
TEXT_COLOR = WHITE          # Màu chữ trên nút/ô số
INFO_TEXT_COLOR = BLACK     # Màu chữ thông tin kết quả
TITLE_COLOR = (50, 50, 50)     # Màu tiêu đề
PATH_TEXT_COLOR = (30, 30, 30) # Màu chữ trong khu vực đường đi
PATH_BG_COLOR = (225, 225, 225) # Màu nền khu vực đường đi
SCROLLBAR_BG_COLOR = (200, 200, 200) # Màu nền thanh cuộn
SCROLLBAR_HANDLE_COLOR = (130, 130, 130) # Màu tay cầm thanh cuộn
SCROLLBAR_HANDLE_HOVER_COLOR = (100, 100, 100) # Màu tay cầm khi di chuột qua

# --- Tham số Thuật toán Di truyền (GA) ---
GA_POPULATION_SIZE = 100     # Kích thước quần thể
GA_MAX_GENERATIONS = 150     # Số thế hệ tối đa
GA_ELITISM_COUNT = 5         # Số cá thể ưu tú giữ lại
GA_TOURNAMENT_SIZE = 5       # Kích thước giải đấu chọn lọc
GA_MUTATION_RATE = 0.15      # Tỷ lệ đột biến
GA_MAX_SEQUENCE_LENGTH = 60  # Độ dài chuỗi nước đi tối đa (giảm để có thể hội tụ nhanh hơn)

# Kích thước màn hình và lưới
SCREEN_WIDTH = 1350
SCREEN_HEIGHT = 650
GRID_SIZE = 3                # Kích thước N (NxN puzzle)
TILE_SIZE = 100              # Kích thước mỗi ô (pixel)
TILE_BORDER_WIDTH = 1        # Độ rộng đường viền ô
PADDING = 15                 # Khoảng đệm chung

# Vị trí các bàn cờ
BOARD_TOTAL_WIDTH = GRID_SIZE * TILE_SIZE
BOARD_TOTAL_HEIGHT = BOARD_TOTAL_WIDTH
BOARD_OFFSET_X_START = PADDING * 3  # Vị trí X bàn cờ bắt đầu
BOARD_OFFSET_Y = PADDING * 5        # Vị trí Y chung cho cả hai bàn cờ
BOARD_SPACING_X = PADDING * 4       # Khoảng cách giữa hai bàn cờ
BOARD_OFFSET_X_GOAL = BOARD_OFFSET_X_START + BOARD_TOTAL_WIDTH + BOARD_SPACING_X # Vị trí X bàn cờ đích

# Kích thước và vị trí nút bấm
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_PADDING = 8           # Khoảng cách giữa các nút
BUTTON_BORDER_RADIUS = 10    # Bo góc nút
BUTTON_START_X_COL1 = BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH + PADDING * 4 # Vị trí X cột nút 1
COL_SPACING = PADDING * 2       # Khoảng cách giữa các cột nút
BUTTON_START_X_COL2 = BUTTON_START_X_COL1 + BUTTON_WIDTH + COL_SPACING      # Vị trí X cột nút 2
BUTTON_COL_START_Y = BOARD_OFFSET_Y # Vị trí Y bắt đầu của hàng nút đầu tiên

# --- Khu vực Hiển thị Đường đi & Thanh cuộn ---
SCROLLBAR_WIDTH = 15          # Độ rộng thanh cuộn
PATH_AREA_X_START = BUTTON_START_X_COL2 + BUTTON_WIDTH + PADDING * 4 # Vị trí X khu vực đường đi
PATH_AREA_Y_START = BOARD_OFFSET_Y # Vị trí Y khu vực đường đi
PATH_AREA_WIDTH = SCREEN_WIDTH - PATH_AREA_X_START - PADDING * 2 - SCROLLBAR_WIDTH - PADDING // 2 # Chiều rộng khu vực đường đi
PATH_AREA_HEIGHT = SCREEN_HEIGHT - PATH_AREA_Y_START - PADDING * 2 # Chiều cao khu vực đường đi
PATH_LINE_SPACING = 5        # Khoảng cách giữa các dòng trong đường đi

# Vị trí thanh cuộn
SCROLLBAR_X = PATH_AREA_X_START + PATH_AREA_WIDTH + PADDING // 2
SCROLLBAR_Y = PATH_AREA_Y_START
SCROLLBAR_HEIGHT = PATH_AREA_HEIGHT

# Font chữ
FONT_NAME = "Arial"
FONT_SIZE_TILE = 40
FONT_SIZE_BUTTON = 16
FONT_SIZE_INFO = 18
FONT_SIZE_TITLE = 22
FONT_SIZE_PATH = 14

# --- Hằng số Cụ thể cho Thuật toán ---
BEAM_WIDTH = 10 # Độ rộng chùm mặc định cho Beam Search

# --- Hằng số Hoạt ảnh ---
MAX_ANIMATION_STEPS_BEFORE_ACCELERATION = 1000 # Ngưỡng để tăng tốc hoạt ảnh
NORMAL_ANIMATION_DELAY = 250 # mili giây mỗi bước cho hoạt ảnh bình thường
FAST_ANIMATION_DELAY = 10    # mili giây mỗi bước cho hoạt ảnh nhanh

# --- Trạng thái Mặc định của Trò chơi ---
# Trạng thái bắt đầu và đích tiêu chuẩn cho 8-puzzle
begin_state = [[1, 8, 2], [0, 4, 3], [7, 6, 5]] # Một trạng thái giải được phổ biến
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]] # Trạng thái đích tiêu chuẩn
# Các nước đi có thể (Lên, Xuống, Trái, Phải) biểu diễn bằng thay đổi (hàng, cột)
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --- Hàm trợ giúp (GA, Thao tác Trạng thái, Heuristics) ---

def find_zero_pos(state):
    """Tìm vị trí (hàng, cột) của ô trống (số 0)."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state[r][c] == 0:
                return r, c
    return -1, -1 # Không nên xảy ra với trạng thái 8-puzzle hợp lệ

def find_tile_pos(num, target_state):
    """Tìm vị trí đích (hàng, cột) của một ô số cụ thể trong trạng thái đích."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if target_state[r][c] == num:
                return r, c
    return -1, -1 # Không nên xảy ra nếu num là 1-8 và target_state hợp lệ

def manhattan_distance(state, target_state):
    """Tính heuristic khoảng cách Manhattan cho một trạng thái."""
    dist = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            num = state[r][c]
            if num != 0: # Không tính khoảng cách cho ô trống
                goal_r, goal_c = find_tile_pos(num, target_state)
                if goal_r != -1: # Đảm bảo ô được tìm thấy trong trạng thái đích
                    dist += abs(r - goal_r) + abs(c - goal_c)
    return dist

def get_neighbors(state):
    """Tạo ra tất cả các trạng thái kế tiếp hợp lệ từ trạng thái hiện tại."""
    neighbors = []
    zero_r, zero_c = find_zero_pos(state)
    if zero_r == -1: return [] # Kiểm tra an toàn

    for dr, dc in moves:
        next_r, next_c = zero_r + dr, zero_c + dc

        # Kiểm tra xem vị trí mới có nằm trong lưới không
        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[zero_r][zero_c], new_state[next_r][next_c] = \
                new_state[next_r][next_c], new_state[zero_r][zero_c]
            neighbors.append(new_state)
    return neighbors


# --- Các hàm Cụ thể cho Thuật toán Di truyền (GA) ---

def apply_move_sequence(start_state, move_sequence):
    """Áp dụng một chuỗi các nước đi ('UP', 'DOWN', 'LEFT', 'RIGHT') vào một trạng thái."""
    current_state = copy.deepcopy(start_state)
    path = [current_state]
    possible_moves_map = {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}
    reverse_moves_map = {v: k for k, v in possible_moves_map.items()} # Ánh xạ 'UP' -> (-1, 0), v.v.

    for move_str in move_sequence:
        zero_r, zero_c = find_zero_pos(current_state)
        if zero_r == -1: break # Trạng thái không hợp lệ

        move_delta = reverse_moves_map.get(move_str)
        if move_delta is None: continue # Chuỗi nước đi không hợp lệ

        dr, dc = move_delta
        next_r, next_c = zero_r + dr, zero_c + dc

        # Kiểm tra xem nước đi có hợp lệ không
        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            new_state = copy.deepcopy(current_state)
            # Thực hiện hoán đổi
            new_state[zero_r][zero_c], new_state[next_r][next_c] = new_state[next_r][next_c], new_state[zero_r][zero_c]
            current_state = new_state
            path.append(current_state)

    return current_state, path

def calculate_fitness(state, target_state):
    """Hàm đánh giá độ thích nghi cho GA - càng thấp càng tốt (càng gần đích)."""
    return manhattan_distance(state, target_state)

def create_random_individual():
    """Tạo một chuỗi nước đi ngẫu nhiên làm cá thể GA."""
    moves_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return [random.choice(moves_list) for _ in range(GA_MAX_SEQUENCE_LENGTH)]

def tournament_selection(population_with_fitness, k):
    """Chọn cá thể tốt nhất từ một giải đấu ngẫu nhiên."""
    selected_tournament = random.sample(population_with_fitness, k)
    selected_tournament.sort(key=lambda x: x[1])
    return selected_tournament[0][0]

def single_point_crossover(parent1, parent2):
    """Thực hiện lai ghép một điểm giữa hai chuỗi cha mẹ."""
    if not isinstance(parent1, list) or not isinstance(parent2, list) or len(parent1) < 2 or len(parent2) < 2:
         return parent1[:], parent2[:]

    min_len = min(len(parent1), len(parent2))
    if min_len < 2: return parent1[:], parent2[:]

    point = random.randint(1, min_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate_sequence(sequence, mutation_rate):
    """Đột biến một chuỗi nước đi bằng cách thay đổi ngẫu nhiên một số nước đi."""
    mutated_sequence = list(sequence) # Làm việc trên bản sao
    moves_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            current_move = mutated_sequence[i]
            possible_new_moves = [m for m in moves_list if m != current_move]
            new_move = random.choice(possible_new_moves if possible_new_moves else moves_list)
            mutated_sequence[i] = new_move
    return mutated_sequence

def GeneticAlgorithm(initial_state, target_state):
    """Thực hiện Thuật toán Di truyền để tìm đường đi giải quyết."""
    print(f"Đang chạy Genetic Algorithm: Pop={GA_POPULATION_SIZE}, Gens={GA_MAX_GENERATIONS}, SeqLen={GA_MAX_SEQUENCE_LENGTH}, MutRate={GA_MUTATION_RATE}")
    start_timer = time.time()
    population = [create_random_individual() for _ in range(GA_POPULATION_SIZE)]
    best_overall_individual = None
    best_overall_fitness = float('inf')
    best_path_found = [initial_state]

    for generation in range(GA_MAX_GENERATIONS):
        population_with_fitness = []
        goal_found = False
        solution_path = None

        for individual in population:
            final_state, _ = apply_move_sequence(initial_state, individual)
            fitness = calculate_fitness(final_state, target_state)
            population_with_fitness.append((individual, fitness))

            if fitness == 0:
                print(f"GA tìm thấy trạng thái đích ở thế hệ {generation}!")
                _, solution_path = apply_move_sequence(initial_state, individual)
                goal_found = True
                break

        if goal_found:
            end_timer = time.time()
            print(f"Thời gian GA: {end_timer - start_timer:.4f} giây")
            return solution_path

        population_with_fitness.sort(key=lambda x: x[1])
        current_best_individual, current_best_fitness = population_with_fitness[0]
        if current_best_fitness < best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual

        if generation % 30 == 0 or generation == GA_MAX_GENERATIONS - 1:
             print(f"Thế hệ {generation}: Fitness tốt nhất={current_best_fitness}, Tổng thể tốt nhất={best_overall_fitness}")

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
    print(f"GA kết thúc sau {GA_MAX_GENERATIONS} thế hệ. Fitness tốt nhất: {best_overall_fitness}. Thời gian GA: {end_timer - start_timer:.4f} giây")
    if best_overall_individual:
         final_state_of_best, best_path_found = apply_move_sequence(initial_state, best_overall_individual)
         final_fitness = calculate_fitness(final_state_of_best, target_state)
         print(f"Trả về đường đi của cá thể tốt nhất (dài {len(best_path_found)-1}). Fitness cuối cùng: {final_fitness}")
         return best_path_found
    else:
        print("GA: Không tìm thấy cá thể phù hợp.")
        return [initial_state]

# --- Thuật toán Tìm kiếm ---

def BFS(start_state, target_state):
    """Breadth-First Search (Tìm kiếm theo chiều rộng)."""
    print("Đang chạy BFS...")
    queue = deque([(start_state, [start_state])])
    visited = {str(start_state)}

    while queue:
        current_state, path = queue.popleft()
        if current_state == target_state:
            return path
        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            if state_str not in visited:
                visited.add(state_str)
                queue.append((neighbor_state, path + [neighbor_state]))
    print("BFS: Không tìm thấy lời giải.")
    return None

def UCS(initial_state, target_state):
    """Uniform Cost Search (Tìm kiếm chi phí đồng nhất - tương đương BFS với chi phí đơn vị)."""
    print("Đang chạy UCS...")
    # Hàng đợi ưu tiên lưu (chi phí, trạng thái, đường đi)
    pq = [(0, initial_state, [initial_state])]
    visited = {str(initial_state): 0} # {chuỗi_trạng_thái: min_cost_để_đạt_được}

    while pq:
        cost, current_state, path = heapq.heappop(pq)
        if current_state == target_state:
            return path
        # Nếu đã tìm thấy đường đi rẻ hơn đến trạng thái này, bỏ qua
        if cost > visited.get(str(current_state), float('inf')):
            continue
        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            new_cost = cost + 1 # Chi phí mỗi nước đi là 1
            if new_cost < visited.get(state_str, float('inf')):
                visited[state_str] = new_cost
                heapq.heappush(pq, (new_cost, neighbor_state, path + [neighbor_state]))
    print("UCS: Không tìm thấy lời giải.")
    return None

def DFS(initial_state, target_state):
    """Depth-First Search (Tìm kiếm theo chiều sâu)."""
    print("Đang chạy DFS...")
    stack = [(initial_state, [initial_state])]
    visited = {str(initial_state)}

    while stack:
        current_state, path = stack.pop()
        if current_state == target_state:
            return path
        # Đảo ngược thứ tự lân cận để khám phá theo thứ tự nhất quán (tùy chọn)
        for neighbor_state in reversed(get_neighbors(current_state)):
            state_str = str(neighbor_state)
            if state_str not in visited:
                visited.add(state_str)
                stack.append((neighbor_state, path + [neighbor_state]))
    print("DFS: Không tìm thấy lời giải.")
    return None

def depth_limited_dfs(start_state, target_state, depth_limit):
    """Thực hiện DFS đến một giới hạn độ sâu cụ thể."""
    stack = [(start_state, [start_state], 0)] # (trạng thái, đường đi, độ sâu)
    visited_depth = {str(start_state): 0} # {chuỗi_trạng_thái: min_depth_để_đạt_được}

    while stack:
        current_state, path, depth = stack.pop()
        if current_state == target_state:
            return path
        if depth < depth_limit:
            for neighbor_state in reversed(get_neighbors(current_state)):
                state_str = str(neighbor_state)
                new_depth = depth + 1
                # Chỉ thêm vào stack nếu chưa thăm hoặc tìm thấy đường đi nông hơn
                if state_str not in visited_depth or new_depth < visited_depth[state_str]:
                     visited_depth[state_str] = new_depth
                     stack.append((neighbor_state, path + [neighbor_state], new_depth))
    return None # Không tìm thấy trong giới hạn độ sâu

def IDDFS(initial_state, target_state):
    """Iterative Deepening Depth-First Search (Tìm kiếm sâu dần lặp)."""
    print("Đang chạy IDDFS...")
    depth_limit = 0
    max_practical_depth = 31 # Giới hạn thực tế để tránh chạy quá lâu
    while True:
        print(f"IDDFS đang thử độ sâu: {depth_limit}")
        solution = depth_limited_dfs(initial_state, target_state, depth_limit)
        if solution is not None:
            return solution
        # Tăng giới hạn độ sâu cho lần lặp tiếp theo
        depth_limit += 1
        if depth_limit > max_practical_depth:
            print(f"IDDFS đạt đến độ sâu thực tế tối đa ({max_practical_depth}) mà không có lời giải.")
            return None

def A_Star(initial_state, target_state):
    """A* Search (Tìm kiếm A*)."""
    print("Đang chạy A*...")
    pq = PriorityQueue()
    start_h = manhattan_distance(initial_state, target_state)
    # Hàng đợi ưu tiên lưu (f_cost, g_cost, trạng thái, đường đi)
    # f_cost = g_cost + h_cost
    pq.put((start_h, 0, initial_state, [initial_state]))
    # Đã thăm lưu {chuỗi_trạng_thái: min_g_cost_để_đạt_được}
    visited_cost = {str(initial_state): 0}

    while not pq.empty():
        f_n, g_cost, current_state, path = pq.get()

        if current_state == target_state:
            return path # Tìm thấy lời giải

        # Nếu đã tìm thấy đường đi ngắn hơn (g_cost thấp hơn) đến trạng thái này, bỏ qua
        if g_cost > visited_cost.get(str(current_state), float('inf')):
            continue

        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            new_g_cost = g_cost + 1 # Chi phí mỗi nước đi là 1

            # Nếu đường đi này ngắn hơn bất kỳ đường đi trước đó đến neighbor_state
            if new_g_cost < visited_cost.get(state_str, float('inf')):
                 visited_cost[state_str] = new_g_cost
                 h_cost = manhattan_distance(neighbor_state, target_state)
                 f_cost = new_g_cost + h_cost
                 pq.put((f_cost, new_g_cost, neighbor_state, path + [neighbor_state]))

    print("A*: Không tìm thấy lời giải.")
    return None

def Greedy(initial_state, target_state):
    """Greedy Best-First Search (Tìm kiếm Tham lam Tốt nhất Đầu tiên)."""
    print("Đang chạy Greedy Best-First...")
    pq = PriorityQueue()
    start_h = manhattan_distance(initial_state, target_state)
    # Hàng đợi ưu tiên lưu (h_cost, trạng thái, đường đi)
    pq.put((start_h, initial_state, [initial_state]))
    visited = {str(initial_state)}

    while not pq.empty():
        h_n, current_state, path = pq.get()
        if current_state == target_state:
            return path # Tìm thấy lời giải

        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            if state_str not in visited:
                visited.add(state_str)
                h_cost = manhattan_distance(neighbor_state, target_state)
                pq.put((h_cost, neighbor_state, path + [neighbor_state]))

    print("Greedy: Không tìm thấy lời giải.")
    return None

def depth_limited_A_star(start_state, target_state, f_limit):
    """Thực hiện tìm kiếm A* chỉ khám phá các nút có f_cost <= f_limit."""
    pq = PriorityQueue()
    start_h = manhattan_distance(start_state, target_state)
    pq.put((start_h, 0, start_state, [start_state])) # (f, g, trạng thái, đường đi)
    visited_cost = {str(start_state): 0} # {chuỗi_trạng_thái: min_g_cost}
    min_exceeded_f = float('inf') # Theo dõi f-cost nhỏ nhất đã vượt quá giới hạn

    while not pq.empty():
        f_n, g_cost, current_state, path = pq.get()

        # Nếu f-cost vượt quá giới hạn, ghi lại f-cost đó (nếu nhỏ hơn) và bỏ qua
        if f_n > f_limit:
            min_exceeded_f = min(min_exceeded_f, f_n)
            continue

        if current_state == target_state:
            return path, float('inf') # Trả về lời giải và giá trị vô hạn cho giới hạn tiếp theo

        # Nếu đã tìm thấy đường đi ngắn hơn đến đây, bỏ qua
        if g_cost > visited_cost.get(str(current_state), float('inf')):
            continue

        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            new_g_cost = g_cost + 1

            if new_g_cost < visited_cost.get(state_str, float('inf')):
                 visited_cost[state_str] = new_g_cost
                 h_new = manhattan_distance(neighbor_state, target_state)
                 f_new = new_g_cost + h_new

                 # Chỉ thêm vào hàng đợi nếu f-cost mới không vượt quá giới hạn
                 if f_new <= f_limit:
                     pq.put((f_new, new_g_cost, neighbor_state, path + [neighbor_state]))
                 else:
                     # Nếu vượt quá, cập nhật f-cost nhỏ nhất đã vượt quá
                     min_exceeded_f = min(min_exceeded_f, f_new)

    # Không tìm thấy lời giải trong giới hạn f hiện tại, trả về giới hạn f tiếp theo cần thử
    return None, min_exceeded_f

def IDA(initial_state, target_state):
    """Iterative Deepening A* Search (Tìm kiếm A* Sâu dần Lặp)."""
    print("Đang chạy IDA*...")
    # Bắt đầu với giới hạn f bằng heuristic của trạng thái ban đầu
    f_limit = manhattan_distance(initial_state, target_state)
    max_f_limit_practical = 80 # Giới hạn thực tế để tránh chạy quá lâu

    while True:
        print(f"IDA* đang thử f-limit: {f_limit}")
        solution, next_f_limit = depth_limited_A_star(initial_state, target_state, f_limit)

        # Nếu tìm thấy lời giải, trả về nó
        if solution is not None:
            return solution

        # Nếu giới hạn tiếp theo là vô hạn, nghĩa là đã khám phá hết hoặc không có lời giải
        if next_f_limit == float('inf'):
            print("IDA*: Không tìm thấy lời giải (đã khám phá tất cả trạng thái có thể đạt được hoặc lời giải cần f > giới hạn hiện tại).")
            return None

        # Nếu giới hạn tiếp theo quá lớn, dừng lại
        if next_f_limit > max_f_limit_practical:
            print(f"IDA* f-limit tiếp theo ({next_f_limit}) vượt quá ngưỡng thực tế ({max_f_limit_practical}). Đang dừng.")
            return None

        # Cập nhật giới hạn f cho lần lặp tiếp theo
        f_limit = next_f_limit


# --- Thuật toán Tìm kiếm Cục bộ ---

def SimpleHillClimbing(initial_state, target_state):
    """Simple Hill Climbing (Leo đồi đơn giản): Di chuyển đến lân cận tốt hơn đầu tiên tìm thấy."""
    print("Đang chạy Simple Hill Climbing...")
    current_state = initial_state
    path = [current_state]
    max_steps = 1000 # Ngăn vòng lặp vô hạn

    for _ in range(max_steps):
        if current_state == target_state:
            return path
        current_h = manhattan_distance(current_state, target_state)
        best_neighbor = None
        found_better = False
        neighbors = get_neighbors(current_state)
        random.shuffle(neighbors) # Thêm tính ngẫu nhiên vào việc chọn lân cận đầu tiên

        for neighbor in neighbors:
             neighbor_h = manhattan_distance(neighbor, target_state)
             if neighbor_h < current_h: # Tìm lân cận đầu tiên tốt hơn
                 best_neighbor = neighbor
                 found_better = True
                 break # Di chuyển ngay khi tìm thấy

        if not found_better:
            print("Simple HC: Đạt đến cực đại địa phương hoặc vùng bình nguyên.")
            return path # Bị kẹt

        current_state = best_neighbor
        path.append(current_state)

    print(f"Simple HC: Đạt số bước tối đa ({max_steps}).")
    return path # Trả về đường đi tốt nhất tìm được

def SteepestHillClimbing(initial_state, target_state):
    """Steepest Ascent Hill Climbing (Leo đồi dốc nhất): Di chuyển đến lân cận tốt nhất."""
    print("Đang chạy Steepest Hill Climbing...")
    current_state = initial_state
    path = [current_state]
    max_steps = 1000 # Ngăn vòng lặp vô hạn

    for _ in range(max_steps):
        if current_state == target_state:
            return path
        current_h = manhattan_distance(current_state, target_state)
        best_neighbor = None
        best_h = current_h # Heuristic tốt nhất tìm thấy cho đến nay (phải nhỏ hơn current_h)
        neighbors = get_neighbors(current_state)

        for neighbor in neighbors:
            neighbor_h = manhattan_distance(neighbor, target_state)
            if neighbor_h < best_h: # Tìm lân cận *tốt nhất* (dốc nhất)
                best_h = neighbor_h
                best_neighbor = neighbor

        # Nếu không tìm thấy lân cận nào tốt hơn hoặc bằng hiện tại
        if best_neighbor is None or best_h >= current_h:
            print("Steepest HC: Đạt đến cực đại địa phương hoặc vùng bình nguyên.")
            return path # Bị kẹt

        current_state = best_neighbor
        path.append(current_state)

    print(f"Steepest HC: Đạt số bước tối đa ({max_steps}).")
    return path # Trả về đường đi tốt nhất tìm được

def StochasticHillClimbing(initial_state, target_state):
    """Stochastic Hill Climbing (Leo đồi ngẫu nhiên): Chọn ngẫu nhiên trong số các lân cận tốt hơn."""
    print("Đang chạy Stochastic Hill Climbing...")
    current_state = initial_state
    path = [current_state]
    max_steps = 1000 # Ngăn vòng lặp vô hạn

    for _ in range(max_steps):
        if current_state == target_state:
            return path
        current_h = manhattan_distance(current_state, target_state)
        uphill_neighbors = [] # Danh sách các lân cận tốt hơn
        neighbors = get_neighbors(current_state)

        for neighbor in neighbors:
            neighbor_h = manhattan_distance(neighbor, target_state)
            if neighbor_h < current_h: # Chỉ xem xét các nước đi cải thiện
                uphill_neighbors.append(neighbor)

        if not uphill_neighbors:
            print("Stochastic HC: Đạt đến cực đại địa phương (không có lân cận tốt hơn).")
            return path # Bị kẹt

        # Chọn ngẫu nhiên từ các lân cận tốt hơn
        current_state = random.choice(uphill_neighbors)
        path.append(current_state)

    print(f"Stochastic HC: Đạt số bước tối đa ({max_steps}).")
    return path # Trả về đường đi tốt nhất tìm được


def SimulatedAnnealing(initial_state, target_state):
    """Simulated Annealing (Luyện kim mô phỏng)."""
    print("Đang chạy Simulated Annealing...")
    temp_start = 15.0   # Nhiệt độ ban đầu cao
    temp_end = 0.01     # Nhiệt độ kết thúc thấp
    alpha = 0.98        # Hệ số giảm nhiệt độ (làm nguội)
    max_iter_per_temp = 100 # Số lần thử ở mỗi mức nhiệt độ

    current_state = copy.deepcopy(initial_state)
    current_cost = manhattan_distance(current_state, target_state)
    path = [current_state]
    best_state_overall = current_state # Theo dõi trạng thái tốt nhất từng thấy
    best_cost_overall = current_cost
    temp = temp_start

    while temp > temp_end: # Tiếp tục cho đến khi đủ "nguội"
        iter_count = 0
        while iter_count < max_iter_per_temp:
            iter_count += 1
            if current_state == target_state:
                print(f"SA đạt trạng thái đích ở nhiệt độ {temp:.3f}")
                if not path or path[-1] != current_state: # Đảm bảo trạng thái cuối được thêm
                    path.append(current_state)
                return path

            neighbors = get_neighbors(current_state)
            if not neighbors: continue # Không có nước đi nào
            # Chọn một lân cận ngẫu nhiên
            neighbor_state = random.choice(neighbors)
            neighbor_cost = manhattan_distance(neighbor_state, target_state)
            delta_e = neighbor_cost - current_cost # Thay đổi chi phí (năng lượng)
            accept_move = False

            if delta_e < 0: # Nếu nước đi mới tốt hơn, luôn chấp nhận
                accept_move = True
                # Cập nhật trạng thái tốt nhất tổng thể nếu cần
                if neighbor_cost < best_cost_overall:
                    best_state_overall = copy.deepcopy(neighbor_state)
                    best_cost_overall = neighbor_cost
            elif temp > 1e-9: # Nếu nước đi mới tệ hơn, chấp nhận với một xác suất
                probability = math.exp(-delta_e / temp) # Xác suất Boltzmann
                if random.random() < probability:
                    accept_move = True

            if accept_move:
                current_state = neighbor_state
                current_cost = neighbor_cost
                # Chỉ thêm vào đường đi nếu trạng thái thay đổi đáng kể (tùy chọn, để đường đi ngắn hơn)
                if not path or path[-1] != current_state:
                     path.append(current_state) # Ghi lại bước đi

        temp *= alpha # Giảm nhiệt độ

    print(f"SA kết thúc. Chi phí cuối cùng: {current_cost}, Chi phí tốt nhất tìm thấy: {best_cost_overall}")
    # Đảm bảo trạng thái cuối cùng được thêm vào đường đi
    if not path or path[-1] != current_state:
        path.append(current_state)
    # Cảnh báo nếu trạng thái tốt nhất là đích nhưng trạng thái cuối cùng thì không
    if best_state_overall == target_state and current_state != target_state:
         print("SA Cảnh báo: Đã thăm trạng thái đích nhưng trạng thái cuối cùng khác.")
    # Có thể muốn trả về đường đi dẫn đến best_state_overall thay vì trạng thái cuối cùng
    # Tùy thuộc vào yêu cầu là tìm đường đi hay chỉ trạng thái tốt nhất
    return path

def BeamSearch(initial_state, target_state, beam_width):
    """Beam Search (Tìm kiếm chùm)."""
    print(f"Đang chạy Beam Search (Độ rộng={beam_width})...")
    start_h = manhattan_distance(initial_state, target_state)
    # Chùm lưu (heuristic, trạng thái, đường đi)
    beam = [(start_h, initial_state, [initial_state])]
    visited = {str(initial_state)} # Tránh chu trình và khám phá lại
    max_iterations = 500 # Ngăn chạy quá lâu
    iterations = 0

    while beam and iterations < max_iterations:
        iterations += 1
        possible_next_states = [] # Thu thập tất cả các trạng thái kế tiếp từ chùm hiện tại

        for h_val, current_state, path in beam:
            # Nếu một trạng thái trong chùm là đích, trả về đường đi của nó
            if current_state == target_state:
                print(f"Beam Search tìm thấy lời giải trong {iterations} lần lặp.")
                return path
            # Tạo các lân cận
            for neighbor in get_neighbors(current_state):
                 possible_next_states.append((neighbor, path + [neighbor]))

        processed_in_step = set() # Tránh thêm cùng một trạng thái nhiều lần trong một bước
        candidates = [] # Các ứng cử viên cho chùm tiếp theo
        for neighbor, new_path in possible_next_states:
            state_str = str(neighbor)
            # Chỉ xem xét các trạng thái chưa được thăm và chưa được xử lý trong bước này
            if state_str not in visited and state_str not in processed_in_step:
                 visited.add(state_str)
                 processed_in_step.add(state_str)
                 neighbor_h = manhattan_distance(neighbor, target_state)
                 candidates.append((neighbor_h, neighbor, new_path))

        # Nếu không có ứng cử viên nào, thuật toán kết thúc
        if not candidates:
            print(f"Beam Search: Chùm trống (không có trạng thái kế tiếp chưa thăm) sau {iterations} lần lặp.")
            # Trả về đường đi tốt nhất từ chùm cuối cùng (nếu có)
            beam.sort(key=lambda x: x[0])
            return beam[0][2] if beam else [initial_state]

        # Sắp xếp các ứng cử viên theo heuristic (tốt nhất trước)
        candidates.sort(key=lambda x: x[0])
        # Chọn beam_width ứng cử viên tốt nhất cho chùm tiếp theo
        beam = candidates[:beam_width]

    # Nếu đạt giới hạn lặp hoặc chùm trống
    if iterations >= max_iterations:
        print(f"Beam Search đạt số lần lặp tối đa ({max_iterations}). Trả về đường đi tốt nhất tìm được.")
    elif not beam:
        print("Beam Search: Chùm trống bất ngờ. Trả về trạng thái ban đầu.")
        return [initial_state] # Hoặc None tùy thuộc vào xử lý lỗi mong muốn

    # Trả về đường đi tốt nhất từ chùm cuối cùng
    beam.sort(key=lambda x: x[0])
    return beam[0][2] if beam else [initial_state]

def AND_OR_Search(initial_state, target_state):
    """AND-OR Search (Thực hiện như A* cho loại puzzle này)."""
    # Đối với 8-puzzle, đồ thị không gian trạng thái không có cấu trúc AND/OR tự nhiên.
    # Mọi nước đi từ một trạng thái dẫn đến một trạng thái kế tiếp (OR nodes).
    # Không có tình huống nào cần phải giải quyết nhiều nhánh con đồng thời (AND nodes).
    # Do đó, A* là một thuật toán phù hợp và hiệu quả hơn cho loại vấn đề này.
    print("Đang chạy AND-OR Search (Sử dụng logic A* vì 8-puzzle là OR-graph)...")
    return A_Star(initial_state, target_state) # Chạy A* như một giải pháp thay thế

# --- Thuật toán Tìm kiếm Không gian Niềm tin (Thích ứng từ A*) ---
def BeliefSpaceSearch(initial_state, target_state):
    """Belief Space Search (Thích ứng từ A* cho 8-puzzle)."""
    # Trong ngữ cảnh của 8-puzzle với thông tin hoàn hảo (trạng thái hoàn toàn có thể quan sát),
    # "không gian niềm tin" thực chất thu gọn thành không gian trạng thái thông thường.
    # Không có sự không chắc chắn về trạng thái hiện tại.
    # Do đó, một thuật toán tìm kiếm thông tin như A* là phù hợp.
    print("Đang chạy Belief Space Search (adapted A*)...")
    pq = PriorityQueue()
    start_h = manhattan_distance(initial_state, target_state)
    # Hàng đợi ưu tiên lưu (f_cost, g_cost, trạng thái, đường đi)
    pq.put((start_h, 0, initial_state, [initial_state]))
    # Đã thăm lưu {chuỗi_trạng_thái: min_g_cost_để_đạt_được}
    visited_cost = {str(initial_state): 0}

    while not pq.empty():
        f_n, g_cost, current_state, path = pq.get()

        if current_state == target_state:
            print("Belief Space Search (adapted A*): Tìm thấy đích.")
            return path # Tìm thấy lời giải

        # Nếu đã tìm thấy đường đi ngắn hơn đến trạng thái này, bỏ qua
        if g_cost > visited_cost.get(str(current_state), float('inf')):
            continue

        for neighbor_state in get_neighbors(current_state):
            state_str = str(neighbor_state)
            new_g_cost = g_cost + 1 # Chi phí mỗi nước đi là 1

            # Nếu đường đi này ngắn hơn bất kỳ đường đi trước đó đến neighbor_state
            if new_g_cost < visited_cost.get(state_str, float('inf')):
                 visited_cost[state_str] = new_g_cost
                 h_cost = manhattan_distance(neighbor_state, target_state)
                 f_cost = new_g_cost + h_cost
                 pq.put((f_cost, new_g_cost, neighbor_state, path + [neighbor_state]))

    print("Belief Space Search (adapted A*): Không tìm thấy lời giải.")
    return None


# --- Hàm Vẽ Pygame ---

def draw_board(screen, state, x_offset, y_offset, font_tile):
    """Vẽ trạng thái bàn cờ puzzle."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            num = state[r][c]
            rect_x = x_offset + c * TILE_SIZE
            rect_y = y_offset + r * TILE_SIZE
            rect = pygame.Rect(rect_x, rect_y, TILE_SIZE, TILE_SIZE)

            if num == 0: # Ô trống
                pygame.draw.rect(screen, EMPTY_TILE_BG_COLOR, rect)
            else: # Ô số
                pygame.draw.rect(screen, BLUE, rect)
                text_surf = font_tile.render(str(num), True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)
            # Vẽ viền
            pygame.draw.rect(screen, TILE_BORDER_COLOR, rect, TILE_BORDER_WIDTH)

def draw_button(screen, text, rect, font_button, normal_color, hover_color, active_color, is_active, is_hover):
    """Vẽ một nút bấm với trạng thái di chuột và hoạt động."""
    color = normal_color
    shadow_offset = 2
    # Tạo màu tối hơn một chút cho hiệu ứng bóng đổ đơn giản
    shadow_color = tuple(max(0, c - 40) for c in color[:3])
    shadow_rect = rect.move(shadow_offset, shadow_offset)
    pygame.draw.rect(screen, shadow_color, shadow_rect, border_radius=BUTTON_BORDER_RADIUS)

    # Chọn màu dựa trên trạng thái
    if is_active: color = active_color
    elif is_hover: color = hover_color

    pygame.draw.rect(screen, color, rect, border_radius=BUTTON_BORDER_RADIUS)
    # Vẽ chữ lên nút
    text_surf = font_button.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def display_results(screen, font_info, algo_name, time_taken, steps, message=None, start_y=500, start_x=50, bg_color=WINDOW_BG_COLOR, final_g_cost=None, final_h_cost=None):
    """Hiển thị kết quả (thời gian, số bước, thông báo) trong khu vực được chỉ định."""
    y_pos = start_y
    x_pos = start_x

    # Tính toán vùng cần xóa trước khi vẽ văn bản mới
    clear_width = (BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH) - BOARD_OFFSET_X_START
    clear_height = 130 # Chiều cao ước tính đủ để chứa thông tin
    clear_rect = pygame.Rect(x_pos - PADDING // 2, y_pos - PADDING // 2, clear_width , clear_height)
    pygame.draw.rect(screen, bg_color, clear_rect) # Xóa vùng hiển thị kết quả cũ

    line_height = font_info.get_height() + 3 # Chiều cao dòng cộng thêm khoảng cách nhỏ
    current_y = y_pos

    if message: # Nếu có thông báo lỗi hoặc trạng thái đặc biệt
        lines = message.split('\n')
        max_lines = 5 # Giới hạn số dòng hiển thị để tránh tràn
        for i, line in enumerate(lines[:max_lines]):
            # Cắt bớt dòng quá dài
            display_line = (line[:85] + '...') if len(line) > 88 else line
            text_surf = font_info.render(display_line, True, RED) # Hiển thị lỗi màu đỏ
            screen.blit(text_surf, (x_pos, current_y))
            current_y += line_height
        if len(lines) > max_lines: # Chỉ ra rằng còn nhiều thông báo hơn
             text_surf = font_info.render("...", True, RED)
             screen.blit(text_surf, (x_pos, current_y))
    elif algo_name: # Nếu có kết quả từ thuật toán
        line1 = f"Algorithm: {algo_name}"
        line2 = f"Time: {time_taken:.4f} second"
        line3 = f"Steps (m / g): {steps}"

        text_surf1 = font_info.render(line1, True, INFO_TEXT_COLOR)
        text_surf2 = font_info.render(line2, True, INFO_TEXT_COLOR)
        text_surf3 = font_info.render(line3, True, INFO_TEXT_COLOR)

        screen.blit(text_surf1, (x_pos, current_y)); current_y += line_height
        screen.blit(text_surf2, (x_pos, current_y)); current_y += line_height
        screen.blit(text_surf3, (x_pos, current_y)); current_y += line_height

        # Hiển thị chi phí g và h của trạng thái cuối nếu có
        if final_g_cost is not None and final_h_cost is not None:
             line4 = f"Heuristic Trạng thái cuối (h): {final_h_cost}"
             text_surf4 = font_info.render(line4, True, INFO_TEXT_COLOR)
             screen.blit(text_surf4, (x_pos, current_y)); current_y += line_height

             # Hiển thị f = g + h cho các thuật toán liên quan
             # Bao gồm BeliefSpaceSearch trong danh sách hiển thị f=g+h
             if algo_name in ["A_Star", "IDA", "AND-OR Search", "Belief Space Search"]:
                 line5 = f"Ước tính Tổng chi phí cuối (f=g+h): {final_g_cost + final_h_cost}"
                 text_surf5 = font_info.render(line5, True, INFO_TEXT_COLOR)
                 screen.blit(text_surf5, (x_pos, current_y))


def animate_solution(solution, screen, font_tile, bg_color=WINDOW_BG_COLOR):
    """Tạo hoạt ảnh cho đường đi lời giải trên bàn cờ đích."""
    if not solution or len(solution) <= 1:
        return True # Không có gì để tạo hoạt ảnh hoặc chỉ có trạng thái bắt đầu

    num_steps = len(solution) - 1
    # Điều chỉnh độ trễ hoạt ảnh dựa trên độ dài đường đi
    delay = NORMAL_ANIMATION_DELAY
    if num_steps > MAX_ANIMATION_STEPS_BEFORE_ACCELERATION:
        delay = FAST_ANIMATION_DELAY
        print(f"Hoạt ảnh: Đường đi dài ({num_steps} bước > {MAX_ANIMATION_STEPS_BEFORE_ACCELERATION}). Tăng tốc hoạt ảnh (delay={delay}ms).")

    goal_board_area = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)

    for i, state in enumerate(solution):
        # Vẽ lại bàn cờ đích với trạng thái hiện tại trong đường đi
        pygame.draw.rect(screen, bg_color, goal_board_area) # Xóa vùng bàn cờ đích
        draw_board(screen, state, BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, font_tile)
        pygame.display.update(goal_board_area) # Chỉ cập nhật vùng bàn cờ đích

        # Xử lý sự kiện trong khi đợi để cho phép thoát
        quit_attempt = False
        start_wait = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_wait < delay:
             for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     quit_attempt = True
                     break
             if quit_attempt: break
             pygame.time.delay(5) # Giảm tải CPU trong khi chờ

        if quit_attempt:
            print("Hoạt ảnh bị hủy bởi người dùng.")
            return False # Báo hiệu hoạt ảnh bị hủy
    return True # Hoạt ảnh hoàn thành

def format_state_to_string(state):
    """Định dạng một trạng thái (list của list) thành biểu diễn chuỗi gọn."""
    # Ví dụ: [[1, 2, 3], [4, 5, 6], [7, 8, 0]] -> "1 2 3 / 4 5 6 / 7 8 0"
    return " / ".join(" ".join(map(str, row)) for row in state)

def draw_solution_path(screen, font_path, solution, path_area_rect, scroll_y, title_font):
    """Vẽ các bước của đường đi lời giải trong khu vực văn bản có thể cuộn."""
    # Vẽ nền và viền cho khu vực đường đi
    pygame.draw.rect(screen, PATH_BG_COLOR, path_area_rect)
    pygame.draw.rect(screen, BLACK, path_area_rect, 1)

    # Vẽ tiêu đề cho khu vực đường đi
    path_title_surf = title_font.render("Finish path", True, TITLE_COLOR)
    path_title_rect = path_title_surf.get_rect(centerx=path_area_rect.centerx, top=path_area_rect.top - title_font.get_height() - PADDING // 2)
    screen.blit(path_title_surf, path_title_rect)

    total_content_height = 0
    if not solution: return total_content_height # Không có đường đi để vẽ

    line_height = font_path.get_height()
    line_height_with_spacing = line_height + PATH_LINE_SPACING
    start_draw_x = path_area_rect.left + PADDING // 2 # Vị trí bắt đầu vẽ X
    content_area_top = path_area_rect.top + PADDING // 2 # Vị trí bắt đầu vẽ Y

    # Tính tổng chiều cao nội dung cần thiết để vẽ tất cả các bước
    total_content_height = len(solution) * line_height_with_spacing

    # Tính toán chỉ số dòng đầu tiên và cuối cùng cần vẽ dựa trên vị trí cuộn
    first_visible_line_index = max(0, int(scroll_y / line_height_with_spacing))
    # Tính số dòng có thể hiển thị cộng thêm vài dòng đệm để tránh hiện tượng cắt ở cạnh
    lines_to_render_count = int(path_area_rect.height / line_height_with_spacing) + 2
    last_visible_line_index = min(len(solution), first_visible_line_index + lines_to_render_count)

    # Giới hạn vùng vẽ chỉ trong khu vực đường đi
    screen.set_clip(path_area_rect)

    # Vẽ các dòng văn bản hiển thị (chỉ những dòng trong phạm vi nhìn thấy)
    for i in range(first_visible_line_index, last_visible_line_index):
        state = solution[i]
        state_str = format_state_to_string(state)
        line_text = f"{i}: {state_str}" # Định dạng dòng: "số_bước: trạng_thái"

        # Tính vị trí Y của dòng hiện tại, có tính đến vị trí cuộn
        draw_y = content_area_top + (i * line_height_with_spacing) - scroll_y

        # Bỏ qua việc vẽ nếu dòng nằm hoàn toàn ngoài vùng hiển thị
        if draw_y + line_height < path_area_rect.top or draw_y > path_area_rect.bottom:
            continue

        text_surf = font_path.render(line_text, True, PATH_TEXT_COLOR)
        text_rect = text_surf.get_rect(left=start_draw_x, top=draw_y)
        screen.blit(text_surf, text_rect)

    # Bỏ giới hạn vùng vẽ
    screen.set_clip(None)
    return total_content_height # Trả về tổng chiều cao nội dung

def draw_scrollbar(screen, scrollbar_track_rect, scroll_y, total_content_height, visible_area_height, handle_hover):
    """Vẽ thanh cuộn cho khu vực đường đi."""
    # Chỉ vẽ thanh cuộn nếu nội dung cao hơn khu vực hiển thị
    if total_content_height <= visible_area_height:
        return None # Không cần thanh cuộn

    # Vẽ đường ray thanh cuộn
    pygame.draw.rect(screen, SCROLLBAR_BG_COLOR, scrollbar_track_rect, border_radius=4)

    # Tính chiều cao của tay cầm thanh cuộn dựa trên tỷ lệ nội dung hiển thị
    handle_height_ratio = visible_area_height / total_content_height
    handle_height = max(20, int(scrollbar_track_rect.height * handle_height_ratio)) # Chiều cao tối thiểu
    handle_height = min(handle_height, scrollbar_track_rect.height) # Không cao hơn đường ray

    # Tính phạm vi cuộn của nội dung và phạm vi di chuyển của tay cầm
    scrollable_range = total_content_height - visible_area_height
    track_movement_range = scrollbar_track_rect.height - handle_height

    # Tính vị trí Y của tay cầm dựa trên vị trí cuộn hiện tại
    handle_y = scrollbar_track_rect.top
    if scrollable_range > 0:
        scroll_ratio = scroll_y / scrollable_range
        handle_y += int(scroll_ratio * track_movement_range)

    # Đảm bảo tay cầm không di chuyển ra ngoài đường ray
    handle_y = max(scrollbar_track_rect.top, min(handle_y, scrollbar_track_rect.bottom - handle_height))
    handle_rect = pygame.Rect(scrollbar_track_rect.left, handle_y, scrollbar_track_rect.width, handle_height)

    # Chọn màu tay cầm dựa trên trạng thái di chuột
    handle_color = SCROLLBAR_HANDLE_HOVER_COLOR if handle_hover else SCROLLBAR_HANDLE_COLOR
    pygame.draw.rect(screen, handle_color, handle_rect, border_radius=4)
    return handle_rect # Trả về hình chữ nhật của tay cầm để xử lý nhấp chuột


# --- Vòng lặp Chính của Trò chơi ---
def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Trình Giải 8-Puzzle")
    clock = pygame.time.Clock()

    # Tải font chữ, có xử lý lỗi nếu font không tồn tại
    try:
        font_tile = pygame.font.SysFont(FONT_NAME, FONT_SIZE_TILE, bold=True)
        font_button = pygame.font.SysFont(FONT_NAME, FONT_SIZE_BUTTON)
        font_info = pygame.font.SysFont(FONT_NAME, FONT_SIZE_INFO)
        font_title = pygame.font.SysFont(FONT_NAME, FONT_SIZE_TITLE, bold=True)
        font_path = pygame.font.SysFont(FONT_NAME, FONT_SIZE_PATH)
    except Exception as e:
        print(f"Lỗi tải font: {e}. Sử dụng font mặc định.")
        font_tile = pygame.font.Font(None, FONT_SIZE_TILE + 10) # Font mặc định với kích thước tương đối
        font_button = pygame.font.Font(None, FONT_SIZE_BUTTON + 6)
        font_info = pygame.font.Font(None, FONT_SIZE_INFO + 6)
        font_title = pygame.font.Font(None, FONT_SIZE_TITLE + 8)
        font_path = pygame.font.Font(None, FONT_SIZE_PATH + 4)

    # Biến Trạng thái Trò chơi
    current_initial_state = copy.deepcopy(begin_state) # Trạng thái bắt đầu hiện tại
    current_goal_state = copy.deepcopy(goal_state)     # Trạng thái đích hiện tại
    solution = None             # Lưu đường đi lời giải
    elapsed_time = 0          # Thời gian chạy thuật toán
    num_steps = 0             # Số bước trong lời giải
    active_button_algo = None # Thuật toán đang được chọn
    last_displayed_algo = None# Thuật toán cuối cùng được hiển thị kết quả
    error_message = None        # Lưu thông báo lỗi
    is_animating = False        # Cờ cho biết hoạt ảnh đang chạy
    solving_in_progress = False # Cờ cho biết thuật toán đang chạy
    final_g = None              # Lưu g_cost của trạng thái cuối
    final_h = None              # Lưu h_cost của trạng thái cuối

    # Trạng thái Thanh cuộn
    path_scroll_y = 0           # Vị trí cuộn hiện tại của khu vực đường đi
    path_total_content_height = 0 # Tổng chiều cao nội dung đường đi
    path_visible_height = PATH_AREA_HEIGHT # Chiều cao nhìn thấy của khu vực đường đi
    scrollbar_track_rect = pygame.Rect(SCROLLBAR_X, SCROLLBAR_Y, SCROLLBAR_WIDTH, SCROLLBAR_HEIGHT)
    scrollbar_handle_rect = None # Hình chữ nhật của tay cầm thanh cuộn
    dragging_scrollbar = False   # Cờ cho biết người dùng đang kéo thanh cuộn
    scrollbar_mouse_offset_y = 0 # Độ lệch giữa vị trí chuột và đỉnh tay cầm khi kéo

    # Danh sách các thuật toán và nút bấm tương ứng
    button_texts = [
        "BFS", "UCS", "DFS", "IDDFS", "A_Star", "IDA", "Greedy",
        "Steepest HC", "Simple HC", "Stochastic HC", "SA",
        "Beam Search", "Genetic Algo", "AND-OR Search",
        "Belief Space Search"
    ]
    button_rects = {} # Từ điển lưu hình chữ nhật cho mỗi nút {văn_bản_nút: rect}

    # Tính toán bố cục nút động để xếp thành 2 cột
    num_buttons = len(button_texts)
    cols = 2
    base_rows_per_col = num_buttons // cols
    extra_buttons = num_buttons % cols
    # Tính số nút trong mỗi cột
    col_counts = [base_rows_per_col + 1 if i < extra_buttons else base_rows_per_col for i in range(cols)]
    col1_count = col_counts[0]
    col2_count = col_counts[1] if cols > 1 else 0

    current_col_x = BUTTON_START_X_COL1
    row_in_col = 0
    col_num = 1
    max_rows_this_col = col1_count
    for i, text in enumerate(button_texts):
        # Chuyển sang cột 2 nếu cần
        if col_num == 2 and row_in_col == 0: # Chỉ cập nhật x khi bắt đầu cột 2
            current_col_x = BUTTON_START_X_COL2
            max_rows_this_col = col2_count
        # Tính vị trí nút
        x = current_col_x
        y = BUTTON_COL_START_Y + row_in_col * (BUTTON_HEIGHT + BUTTON_PADDING)
        button_rects[text] = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
        row_in_col += 1
        # Chuyển sang cột tiếp theo nếu đã đủ hàng cho cột hiện tại
        if row_in_col >= max_rows_this_col and col_num < cols:
            col_num += 1
            row_in_col = 0 # Đặt lại số hàng cho cột mới

    # Định vị nút Run và Reset bên dưới các nút thuật toán
    max_rows_used = max(col_counts) if col_counts else 0
    run_reset_y_start = BUTTON_COL_START_Y + max_rows_used * (BUTTON_HEIGHT + BUTTON_PADDING)
    run_button_x = BUTTON_START_X_COL1
    run_button_y = run_reset_y_start + PADDING
    run_rect = pygame.Rect(run_button_x, run_button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    reset_button_x = BUTTON_START_X_COL2
    reset_button_y = run_button_y
    reset_rect = pygame.Rect(reset_button_x, reset_button_y, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Định vị các thành phần Giao diện Người dùng khác
    controls_area_width = (BUTTON_START_X_COL2 + BUTTON_WIDTH) - BUTTON_START_X_COL1
    controls_title_center_x = BUTTON_START_X_COL1 + controls_area_width // 2
    controls_title_y_bottom = BUTTON_COL_START_Y - PADDING
    results_area_x = BOARD_OFFSET_X_START
    results_area_y = BOARD_OFFSET_Y + BOARD_TOTAL_HEIGHT + PADDING * 4
    results_title_y = results_area_y - font_title.get_height() - PADDING // 2
    path_area_rect = pygame.Rect(PATH_AREA_X_START, PATH_AREA_Y_START, PATH_AREA_WIDTH, PATH_AREA_HEIGHT)
    start_board_rect = pygame.Rect(BOARD_OFFSET_X_START, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)
    goal_board_rect = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)

    # Ánh xạ tên nút tới hàm thuật toán tương ứng
    algorithm_map = {
        "BFS": BFS, "UCS": UCS, "DFS": DFS, "IDDFS": IDDFS,
        "A_Star": A_Star, "IDA": IDA, "Greedy": Greedy,
        "Steepest HC": SteepestHillClimbing, "Simple HC": SimpleHillClimbing,
        "Stochastic HC": StochasticHillClimbing, "SA": SimulatedAnnealing,
        # Beam Search cần tham số beam_width
        "Beam Search": lambda start, goal: BeamSearch(start, goal, BEAM_WIDTH),
        "Genetic Algo": GeneticAlgorithm,
        "AND-OR Search": AND_OR_Search,
        "Belief Space Search": BeliefSpaceSearch
    }

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered_button = None       # Nút đang được di chuột qua
        scrollbar_handle_hover = False # Tay cầm thanh cuộn đang được di chuột qua
        events = pygame.event.get() # Lấy tất cả sự kiện từ hàng đợi

        # --- Xử lý Sự kiện Toàn cục (luôn chạy) ---
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break
        if not running: break # Thoát vòng lặp chính nếu nhận tín hiệu thoát

        # --- Xử lý Đầu vào (Chỉ khi không có hoạt ảnh hoặc đang giải) ---
        # Ngăn chặn tương tác người dùng khi thuật toán đang chạy hoặc hoạt ảnh đang diễn ra
        if not is_animating and not solving_in_progress:
            for event in events:
                # Cuộn Chuột để cuộn khu vực đường đi
                if event.type == pygame.MOUSEWHEEL:
                    # Chỉ cuộn nếu chuột ở trên khu vực đường đi và có nội dung để cuộn
                    if path_area_rect.collidepoint(mouse_pos) and path_total_content_height > path_visible_height:
                        scroll_amount = event.y * -30 # event.y là 1 hoặc -1, nhân với hệ số tốc độ cuộn
                        path_scroll_y += scroll_amount
                        # Giới hạn vị trí cuộn trong phạm vi hợp lệ
                        max_scroll = path_total_content_height - path_visible_height
                        path_scroll_y = max(0, min(path_scroll_y, max_scroll))
                # Nhấn Chuột Xuống
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Nút chuột trái
                    clicked_handled = False # Cờ để tránh xử lý nhiều lần một cú nhấp
                    # Kéo Thanh cuộn
                    if scrollbar_handle_rect and scrollbar_handle_rect.collidepoint(event.pos):
                        dragging_scrollbar = True
                        # Lưu độ lệch giữa vị trí nhấp chuột và đỉnh tay cầm
                        scrollbar_mouse_offset_y = event.pos[1] - scrollbar_handle_rect.top
                        clicked_handled = True
                    # Nhấp Đường ray Thanh cuộn (để nhảy đến vị trí)
                    elif scrollbar_track_rect.collidepoint(event.pos) and path_total_content_height > path_visible_height:
                        if scrollbar_handle_rect: # Đảm bảo tay cầm tồn tại
                            scrollable_range = path_total_content_height - path_visible_height
                            handle_h = scrollbar_handle_rect.height
                            track_movement_range = scrollbar_track_rect.height - handle_h
                            if track_movement_range > 0 and scrollable_range > 0:
                                # Tính vị trí cuộn tương ứng với vị trí nhấp trên đường ray
                                relative_y_click = (event.pos[1] - scrollbar_track_rect.top) / scrollbar_track_rect.height
                                target_scroll_y = (relative_y_click * scrollable_range)
                                # Cập nhật và giới hạn vị trí cuộn
                                path_scroll_y = max(0, min(target_scroll_y, scrollable_range))
                                clicked_handled = True
                    # Nhấp Nút Thuật toán
                    if not clicked_handled:
                        for text, rect in button_rects.items():
                            if rect.collidepoint(event.pos):
                                # Nếu chọn thuật toán mới, đặt lại trạng thái kết quả
                                if active_button_algo != text:
                                    solution = None; last_displayed_algo = None; error_message = None
                                    elapsed_time = 0; num_steps = 0; path_scroll_y = 0
                                    final_g = None; final_h = None
                                active_button_algo = text # Đặt thuật toán được chọn
                                print(f"Đã chọn thuật toán: {active_button_algo}")
                                clicked_handled = True
                                break
                        # Nhấp Nút Run
                        if not clicked_handled and run_rect.collidepoint(event.pos):
                            if active_button_algo: # Chỉ chạy nếu đã chọn thuật toán
                                print(f"Đang chạy {active_button_algo}...")
                                # Đặt lại trạng thái trước khi chạy
                                solution = None; error_message = None; last_displayed_algo = None
                                num_steps = 0; elapsed_time = 0; solving_in_progress = True
                                path_scroll_y = 0; final_g = None; final_h = None
                                # Cập nhật hiển thị ngay lập tức để hiển thị "Đang giải..."
                                # Xác định các vùng cần vẽ lại
                                results_clear_area = pygame.Rect(results_area_x - PADDING//2, results_area_y - PADDING//2, (BOARD_OFFSET_X_GOAL + BOARD_TOTAL_WIDTH) - BOARD_OFFSET_X_START, 130)
                                path_clear_area = path_area_rect.inflate(0, font_title.get_height() + PADDING).union(scrollbar_track_rect)
                                # Xóa và vẽ lại các vùng đó
                                screen.fill(WINDOW_BG_COLOR, results_clear_area)
                                screen.fill(WINDOW_BG_COLOR, path_clear_area)
                                display_results(screen, font_info, None, 0, 0, "Solving...", start_y=results_area_y, start_x=results_area_x, bg_color=WINDOW_BG_COLOR)
                                pygame.display.update([results_clear_area, path_clear_area]) # Chỉ cập nhật các vùng bị ảnh hưởng
                            else:
                                error_message = "Vui lòng chọn thuật toán trước."
                                last_displayed_algo = None # Xóa kết quả cũ nếu có
                            clicked_handled = True
                        # Nhấp Nút Reset
                        if not clicked_handled and reset_rect.collidepoint(event.pos):
                            print("Đặt lại trạng thái và lựa chọn.")
                            # Đặt lại tất cả về giá trị ban đầu
                            current_initial_state = copy.deepcopy(begin_state)
                            current_goal_state = copy.deepcopy(goal_state)
                            solution = None; active_button_algo = None; last_displayed_algo = None
                            elapsed_time = 0; num_steps = 0; error_message = None
                            is_animating = False; solving_in_progress = False; path_scroll_y = 0
                            final_g = None; final_h = None
                            clicked_handled = True
                        # Nhấp Bàn cờ Bắt đầu (để di chuyển ô trống)
                        if not clicked_handled and start_board_rect.collidepoint(event.pos):
                            click_x, click_y = event.pos
                            # Chuyển đổi tọa độ pixel sang tọa độ hàng/cột
                            col = (click_x - BOARD_OFFSET_X_START) // TILE_SIZE
                            row = (click_y - BOARD_OFFSET_Y) // TILE_SIZE
                            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                                zero_r, zero_c = find_zero_pos(current_initial_state)
                                # Kiểm tra xem ô được nhấp có kề với ô trống không
                                if zero_r != -1 and abs(row - zero_r) + abs(col - zero_c) == 1:
                                    # Hoán đổi ô được nhấp với ô trống
                                    current_initial_state[zero_r][zero_c], current_initial_state[row][col] = \
                                        current_initial_state[row][col], current_initial_state[zero_r][zero_c]
                                    # Đặt lại kết quả vì trạng thái bắt đầu đã thay đổi
                                    solution = None; last_displayed_algo = None; error_message = None
                                    path_scroll_y = 0; final_g = None; final_h = None
                            clicked_handled = True
                        # Nhấp Bàn cờ Đích (để di chuyển ô trống)
                        if not clicked_handled and goal_board_rect.collidepoint(event.pos):
                            click_x, click_y = event.pos
                            col = (click_x - BOARD_OFFSET_X_GOAL) // TILE_SIZE
                            row = (click_y - BOARD_OFFSET_Y) // TILE_SIZE
                            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                                zero_r, zero_c = find_zero_pos(current_goal_state)
                                if zero_r != -1 and abs(row - zero_r) + abs(col - zero_c) == 1:
                                    # Hoán đổi ô được nhấp với ô trống trong trạng thái đích
                                    current_goal_state[zero_r][zero_c], current_goal_state[row][col] = \
                                        current_goal_state[row][col], current_goal_state[zero_r][zero_c]
                                    print(f"Trạng thái đích được sửa đổi bởi người dùng: {current_goal_state}")
                                    # Đặt lại kết quả vì trạng thái đích đã thay đổi
                                    solution = None; last_displayed_algo = None; error_message = None
                                    path_scroll_y = 0; final_g = None; final_h = None
                            clicked_handled = True
                # Thả Chuột (dừng kéo thanh cuộn)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if dragging_scrollbar: dragging_scrollbar = False
                # Di Chuột (để kéo thanh cuộn)
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_scrollbar and scrollbar_handle_rect:
                        # Tính vị trí mới của đỉnh tay cầm dựa trên chuyển động chuột
                        new_handle_top = event.pos[1] - scrollbar_mouse_offset_y
                        # Giới hạn vị trí tay cầm trong đường ray
                        new_handle_top = max(scrollbar_track_rect.top, min(new_handle_top, scrollbar_track_rect.bottom - scrollbar_handle_rect.height))
                        # Tính vị trí cuộn tương ứng
                        scrollable_range = path_total_content_height - path_visible_height
                        track_movement_range = scrollbar_track_rect.height - scrollbar_handle_rect.height
                        if track_movement_range > 0 and scrollable_range > 0:
                             relative_handle_pos = (new_handle_top - scrollbar_track_rect.top) / track_movement_range
                             path_scroll_y = relative_handle_pos * scrollable_range
                             # Giới hạn vị trí cuộn
                             path_scroll_y = max(0, min(path_scroll_y, scrollable_range))

        # --- Quá trình Giải (Chạy thuật toán) ---
        if solving_in_progress:
            start_time = time.time()
            solver_func = algorithm_map.get(active_button_algo) # Lấy hàm thuật toán từ ánh xạ
            temp_solution = None
            current_error_message = None # Lỗi cụ thể trong lần chạy này
            try:
                if solver_func:
                    print(f"Đang thực thi {active_button_algo}...")
                    temp_solution = solver_func(current_initial_state, current_goal_state) # Gọi hàm giải
                    print(f"Thực thi {active_button_algo} hoàn tất.")
                else:
                    # Lỗi này không nên xảy ra nếu ánh xạ được định nghĩa đúng
                    current_error_message = f"Lỗi Nội bộ: Không tìm thấy hàm thuật toán cho '{active_button_algo}'!"
            except Exception as e:
                # Bắt lỗi runtime trong quá trình thực thi thuật toán
                current_error_message = f"Lỗi Runtime trong {active_button_algo}:\n{type(e).__name__}: {e}"
                print(f"--- Lỗi khi thực thi {active_button_algo} ---")
                traceback.print_exc() # In chi tiết lỗi ra console
                print("------------------------------------------")
            end_time = time.time()
            elapsed_time = end_time - start_time # Tính thời gian chạy
            solution = temp_solution # Lưu kết quả trả về
            error_message = current_error_message # Lưu lỗi nếu có
            is_animating = False # Hoạt ảnh sẽ chỉ bắt đầu nếu có lời giải hợp lệ
            num_steps = 0
            final_g = None; final_h = None
            last_displayed_algo = active_button_algo # Lưu tên thuật toán đã chạy để hiển thị

            if solution: # Nếu thuật toán trả về một đường đi
                num_steps = len(solution) - 1 if len(solution) > 0 else 0
                final_state = solution[-1] # Lấy trạng thái cuối cùng của đường đi
                final_g = num_steps # g_cost của trạng thái cuối là độ dài đường đi
                final_h = manhattan_distance(final_state, current_goal_state) # Tính heuristic của trạng thái cuối

                if final_state == current_goal_state: # Kiểm tra xem có phải là lời giải đúng không
                    error_message = None # Không có lỗi nếu tìm thấy đích
                    print(f"Tìm thấy lời giải: {active_button_algo}, Thời gian: {elapsed_time:.4f}s, Số bước: {num_steps}.")
                    is_animating = True # Bật cờ để bắt đầu hoạt ảnh
                else: # Nếu đường đi không dẫn đến đích
                    error_message = (f"{active_button_algo}: Tìm thấy đường đi (dài {num_steps}), "
                                     f"nhưng kết thúc ở trạng thái H={final_h} (không phải đích).")
                    print(error_message)
                    is_animating = False # Không tạo hoạt ảnh nếu không đến đích
            elif not error_message: # Nếu thuật toán trả về None và không có lỗi runtime
                error_message = f"{active_button_algo}: Không tìm thấy hoặc trả về đường đi lời giải."
                print(error_message)
                num_steps = 0

            path_scroll_y = 0 # Đặt lại vị trí cuộn khi có kết quả mới
            solving_in_progress = False # Kết thúc quá trình giải

        # --- Cập nhật Trạng thái Di chuột (chỉ khi không giải/hoạt ảnh) ---
        hovered_button = None
        scrollbar_handle_hover = False
        if not is_animating and not solving_in_progress:
            # Kiểm tra di chuột qua tay cầm thanh cuộn trước
            if scrollbar_handle_rect and scrollbar_handle_rect.collidepoint(mouse_pos):
                scrollbar_handle_hover = True
            else:
                # Kiểm tra di chuột qua các nút thuật toán
                for text, rect in button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        hovered_button = text
                        break
                # Kiểm tra di chuột qua nút Run/Reset nếu chưa có nút nào khác được hover
                if not hovered_button:
                    if run_rect.collidepoint(mouse_pos): hovered_button = "Run"
                    elif reset_rect.collidepoint(mouse_pos): hovered_button = "Reset"

        # --- Vẽ (Chỉ vẽ lại toàn bộ màn hình khi không có hoạt ảnh) ---
        if not is_animating:
            screen.fill(WINDOW_BG_COLOR) # Xóa màn hình bằng màu nền

            # Vẽ Tiêu đề cho các khu vực
            start_title_surf = font_title.render("Begin State", True, TITLE_COLOR)
            goal_title_surf = font_title.render("Goal State", True, TITLE_COLOR)
            controls_title_surf = font_title.render("Controller", True, TITLE_COLOR)
            results_title_surf = font_title.render("Result", True, TITLE_COLOR)
            # Vẽ tiêu đề bàn cờ
            screen.blit(start_title_surf, (BOARD_OFFSET_X_START, PADDING * 2))
            screen.blit(goal_title_surf, (BOARD_OFFSET_X_GOAL, PADDING * 2))
            # Vẽ tiêu đề khu vực điều khiển (căn giữa)
            controls_title_rect = controls_title_surf.get_rect(centerx=controls_title_center_x, bottom=controls_title_y_bottom)
            screen.blit(controls_title_surf, controls_title_rect)
            # Vẽ tiêu đề khu vực kết quả
            screen.blit(results_title_surf, (results_area_x, results_title_y))

            # Vẽ Bàn cờ Bắt đầu
            draw_board(screen, current_initial_state, BOARD_OFFSET_X_START, BOARD_OFFSET_Y, font_tile)

            # Xác định trạng thái để hiển thị trên Bàn cờ Đích
            display_state_on_goal = current_goal_state # Mặc định hiển thị trạng thái đích
            if solution and not is_animating: # Nếu có lời giải và không đang hoạt ảnh
                 # Hiển thị trạng thái cuối cùng của đường đi tìm được trên bàn cờ đích
                display_state_on_goal = solution[-1]
            draw_board(screen, display_state_on_goal, BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, font_tile)

            # Vẽ các Nút thuật toán
            for text, rect in button_rects.items():
                is_active = (active_button_algo == text) # Nút đang được chọn
                is_hover = (hovered_button == text)      # Nút đang được di chuột qua
                draw_button(screen, text, rect, font_button, STEEL_BLUE, LIGHT_BLUE, ORANGE, is_active, is_hover)
            # Vẽ nút Run và Reset
            draw_button(screen, "Run", run_rect, font_button, GREEN, LIGHT_BLUE, ORANGE, False, hovered_button == "Run")
            draw_button(screen, "Reset", reset_rect, font_button, RED, LIGHT_BLUE, ORANGE, False, hovered_button == "Reset")

            # Hiển thị Kết quả (thời gian, bước, lỗi, g/h cuối)
            display_results(screen, font_info, last_displayed_algo, elapsed_time, num_steps, error_message,
                            start_y=results_area_y, start_x=results_area_x, bg_color=WINDOW_BG_COLOR,
                            final_g_cost=final_g, final_h_cost=final_h)

            # Vẽ Đường đi & Thanh cuộn
            path_total_content_height = draw_solution_path(screen, font_path, solution, path_area_rect, path_scroll_y, font_title)
            scrollbar_handle_rect = draw_scrollbar(screen, scrollbar_track_rect, path_scroll_y, path_total_content_height, path_visible_height, scrollbar_handle_hover or dragging_scrollbar)

            pygame.display.flip() # Cập nhật toàn bộ màn hình

        # --- Bước Hoạt ảnh (Chỉ chạy nếu is_animating là True) ---
        if is_animating:
            print("Bắt đầu hoạt ảnh lời giải...")
            # Chỉ tạo hoạt ảnh nếu giải pháp thực sự đạt được mục tiêu
            animation_success = False
            if solution and solution[-1] == current_goal_state:
                # Vẽ trạng thái ban đầu của bàn cờ đích trước khi bắt đầu hoạt ảnh
                # Điều này đảm bảo hoạt ảnh bắt đầu từ trạng thái ban đầu, không phải trạng thái đích ngay lập tức
                draw_board(screen, solution[0], BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, font_tile)
                goal_board_area = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)
                pygame.display.update(goal_board_area) # Cập nhật chỉ vùng bàn cờ đích
                pygame.time.delay(100) # Tạm dừng ngắn để người dùng thấy trạng thái bắt đầu

                # Chạy hàm tạo hoạt ảnh
                animation_success = animate_solution(solution, screen, font_tile, bg_color=WINDOW_BG_COLOR)
                if not animation_success:
                    # Nếu người dùng thoát trong lúc hoạt ảnh, dừng vòng lặp chính
                    running = False
            else:
                 # Nếu lời giải không đạt đích (ví dụ: Hill Climbing bị kẹt)
                 print("Hoạt ảnh bị bỏ qua vì lời giải không đạt được trạng thái đích.")
                 # Trực quan hiển thị trạng thái cuối cùng trên bàn cờ đích mà không tạo hoạt ảnh
                 if solution:
                     draw_board(screen, solution[-1], BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, font_tile)
                     goal_board_area = pygame.Rect(BOARD_OFFSET_X_GOAL, BOARD_OFFSET_Y, BOARD_TOTAL_WIDTH, BOARD_TOTAL_HEIGHT)
                     pygame.display.update(goal_board_area)


            is_animating = False # Hoạt ảnh kết thúc (hoặc bị bỏ qua)

        clock.tick(60) # Giới hạn tốc độ khung hình

    pygame.quit() # Dọn dẹp Pygame khi thoát
    print("Pygame đã đóng thành công.")

if __name__ == "__main__":
    main()
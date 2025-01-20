import pygame
import numpy as np
import math
import heapq
import matplotlib

# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import scipy.ndimage as ndi  # 引入scipy.ndimage
from scipy.interpolate import splprep, splev  # 引入样条插值所需的函数

# -------------------- 配置日志 --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
SAFE_DISTANCE = 10  # 安全距离（像素）
OBSTACLE_BUFFER = 15  # 障碍物安全缓冲区
GOAL_THRESHOLD = 10  # 到达目标的阈值（像素）
SMOOTH_FACTOR = 300  # 路径平滑因子

lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

SHIP_RADIUS = 15  # 船舶的半径，用于路径规划

# -------------------- 路径距离和转向点统计函数 --------------------
def compute_path_distance(path):
    """
    计算一条路径的总几何长度（逐段欧几里得距离之和）。
    path 为 [(y0, x0), (y1, x1), ...]
    """
    total_dist = 0.0
    for i in range(len(path) - 1):
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        seg_dist = math.hypot(x2 - x1, y2 - y1)
        total_dist += seg_dist
    return total_dist

def count_turns_by_angle(path, angle_threshold_degrees=15):
    """
    统计路径中方向变化大于指定角度阈值的转向点数量。
    path: [(y0, x0), (y1, x1), (y2, x2), ...]
    当两段向量的夹角 >= angle_threshold_degrees 时，视为一次转向。
    """
    if len(path) < 3:
        return 0

    turn_count = 0
    for i in range(1, len(path) - 1):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        y2, x2 = path[i + 1]

        # 两段向量 v1, v2
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)

        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        # 避免零长度向量的除 0 问题
        if mag1 < 1e-9 or mag2 < 1e-9:
            continue

        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_val = dot / (mag1 * mag2)
        # 数值保护，防止浮点误差导致 cos_val 超出 [-1, 1]
        cos_val = max(-1.0, min(1.0, cos_val))
        angle_degrees = math.degrees(math.acos(cos_val))

        if angle_degrees >= angle_threshold_degrees:
            turn_count += 1

    return turn_count

# -------------------- 辅助函数 --------------------
def generate_map_surface():
    fig = plt.figure(figsize=(WINDOW_WIDTH / 100, WINDOW_HEIGHT / 100), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.set_title("Path Planning Using FMM Algorithm", fontsize=16)

    ax.set_xticks(np.arange(lon_min, lon_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{lon:.1f}°E" for lon in np.arange(lon_min, lon_max + 0.1, 0.2)], fontsize=10)
    ax.set_yticklabels([f"{lat:.1f}°N" for lat in np.arange(lat_min, lat_max + 0.1, 0.2)], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    surface = pygame.image.frombuffer(buf[:, :, :3].tobytes(), (w, h), "RGB")  # 只取RGB部分
    return surface, buf[:, :, :3]

def create_environment_from_array(arr):
    h, w, _ = arr.shape
    env = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            r, g, b = arr[y, x]
            if b > r and b > g:
                env[y, x] = 0  # 海洋区域
            else:
                env[y, x] = 1  # 陆地或障碍物
    return env

def is_valid_position(x, y, environment, width, height):
    if not (0 <= x < width and 0 <= y < height):
        return False
    if x < 20 or x > width - 20 or y < 20 or y > height - 20:
        return False
    if environment[int(y), int(x)] == 1:
        return False
    return True

def latlon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, w, h):
    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h
    return x, y

def inflate_obstacles(env, distance):
    """
    对环境中的障碍物（陆地）进行膨胀，确保路径规划时保持安全距离。
    """
    structure = ndi.generate_binary_structure(2, 1)  # 4连通结构
    env = env.copy()
    env = ndi.binary_dilation(env, structure=structure, iterations=distance).astype(env.dtype)
    return env

# -------------------- 路径规划 (FMM) --------------------
def fmm(env, start, goal):
    """
    使用简单的 Dijkstra 思路在 8 邻域上进行传播，
    这里称作 FMM 仅做示例，实际上也可以视为 BFS / Dijkstra。
    """
    rows, cols = env.shape
    dist = np.full((rows, cols), np.inf)
    dist[start[0], start[1]] = 0
    pq = [(0, start)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    visited = np.zeros((rows, cols), dtype=bool)

    while pq:
        d, (cy, cx) = heapq.heappop(pq)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        if (cy, cx) == goal:
            break
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols and env[ny, nx] == 0:
                nd = d + math.sqrt(dy**2 + dx**2)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, (ny, nx)))

    # 若无法到达 goal，dist[goal] 会是 inf
    if dist[goal] == np.inf:
        return []

    # 反向回溯路径
    path = []
    current = goal
    while current != start:
        path.append(current)
        cy, cx = current
        candidates = [(cy + dy, cx + dx) for dy, dx in dirs
                      if 0 <= cy + dy < rows and 0 <= cx + dx < cols]
        if not candidates:
            break
        # 找到代价最小的候选
        current = min(candidates, key=lambda c: dist[c])
    path.append(start)
    path.reverse()
    return path

# -------------------- 标记障碍物 --------------------
def mark_obstacle(env, x, y, size, inflate):
    total_size = size + inflate
    for dy in range(-total_size, total_size + 1):
        for dx in range(-total_size, total_size + 1):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < env.shape[0] and 0 <= nx < env.shape[1]:
                if math.sqrt(dx**2 + dy**2) <= total_size:
                    env[ny, nx] = 1

def unmark_obstacle(env, x, y, size, inflate=0):
    total_size = size + inflate
    for dy in range(-total_size, total_size + 1):
        for dx in range(-total_size, total_size + 1):
            ny = int(y) + dy
            nx = int(x) + dx
            if 0 <= ny < env.shape[0] and 0 <= nx < env.shape[1]:
                if math.sqrt(dx**2 + dy**2) <= total_size:
                    env[ny, nx] = 0

# -------------------- 类定义 --------------------
class Obstacle:
    def __init__(self, x, y, is_dynamic=False, size=5, shape='circle', color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.is_dynamic = is_dynamic
        self.size = size
        self.shape = shape
        self.color = color

    def draw(self, win):
        if self.shape == 'circle':
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.size)
        elif self.shape == 'square':
            pygame.draw.rect(win, self.color,
                             (int(self.x) - self.size, int(self.y) - self.size, self.size * 2, self.size * 2))
        elif self.shape == 'triangle':
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            pygame.draw.polygon(win, self.color, points)

# -------------------- 绘图函数 --------------------
def draw_path(win, path, color=(0, 0, 255), width=2):
    if len(path) > 1:
        pygame.draw.lines(win, color, False, path, width)

def draw_star(win, color, position, size):
    x, y = position
    points = []
    for i in range(5):
        angle = i * 2 * math.pi / 5 - math.pi / 2
        outer = (x + size * math.cos(angle), y + size * math.sin(angle))
        inner_angle = angle + math.pi / 5
        inner = (x + size/2 * math.cos(inner_angle), y + size/2 * math.sin(inner_angle))
        points.extend([outer, inner])
    pygame.draw.polygon(win, color, points)

# -------------------- 路径平滑函数 --------------------
def smooth_path(path, smooth_factor=300):
    """
    使用样条插值对路径进行平滑处理。
    参数：
        path (list of tuples): 原始路径，包含 (y, x) 点。
        smooth_factor (float): 平滑因子，越大平滑程度越高。
    返回：
        list of tuples: 平滑后的路径，依旧是 (y, x) 格式。
    """
    if len(path) < 3:
        return path
    y, x = zip(*path)
    try:
        tck, u = splprep([x, y], s=smooth_factor)
        num_points = max(100, len(path))
        u_new = np.linspace(0, 1, num=num_points)
        x_new, y_new = splev(u_new, tck)
        smoothed_path = list(zip(y_new, x_new))
        return smoothed_path
    except Exception as e:
        logging.error(f"路径平滑失败: {e}")
        return path

# -------------------- 可放置障碍物检测 --------------------
def can_place_obstacle(x, y, obstacles, environment, new_size, buffer_distance=SAFE_DISTANCE):
    """
    检查是否可以在 (x, y) 放置障碍物，不与现有障碍物重叠或过近
    """
    if environment[int(y), int(x)] == 1:
        return False
    for obs in obstacles:
        distance = math.hypot(x - obs.x, y - obs.y)
        if distance < (obs.size + new_size + buffer_distance):
            return False
    return True

# -------------------- 主程序 --------------------
def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Path Planning Using FMM Algorithm")
    clock = pygame.time.Clock()

    # 1. 环境初始化
    background, bg_array = generate_map_surface()
    height, width, _ = bg_array.shape
    environment_static = create_environment_from_array(bg_array)

    # 2. 膨胀陆地以确保安全距离
    environment_static = inflate_obstacles(environment_static, SAFE_DISTANCE)

    # 3. 设置起点和终点
    fixed_start_lat, fixed_start_lon = 12.2, 123.3
    fixed_goal_lat, fixed_goal_lon = 12.5, 123.8
    start_x, start_y = latlon_to_pixel(fixed_start_lat, fixed_start_lon,
                                       lat_min, lat_max, lon_min, lon_max,
                                       width, height)
    goal_x, goal_y = latlon_to_pixel(fixed_goal_lat, fixed_goal_lon,
                                     lat_min, lat_max, lon_min, lon_max,
                                     width, height)
    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # 确保起点和终点周围有足够的空间
    buffer_size = SAFE_DISTANCE + SHIP_RADIUS
    for dy in range(-buffer_size, buffer_size + 1):
        for dx in range(-buffer_size, buffer_size + 1):
            ny, nx = int(start_y) + dy, int(start_x) + dx
            if 0 <= ny < height and 0 <= nx < width:
                environment_static[ny, nx] = 0

    for dy in range(-buffer_size, buffer_size + 1):
        for dx in range(-buffer_size, buffer_size + 1):
            ny, nx = int(goal_y) + dy, int(goal_x) + dx
            if 0 <= ny < height and 0 <= nx < width:
                environment_static[ny, nx] = 0

    # 4. 生成静态障碍物
    static_obstacle_positions = [
        (580, 397), (294, 159), (554, 344), (435, 194), (216, 117), (395, 453),
        (260, 434), (371, 485), (197, 151), (225, 285), (394, 229), (202, 209),
        (555, 233), (246, 367), (276, 174), (392, 173), (273, 353), (191, 261),
        (568, 280), (468, 181)
    ]
    static_obstacles = []
    for pos in static_obstacle_positions:
        x, y = pos
        if can_place_obstacle(x, y, static_obstacles, environment_static, new_size=5):
            for dy in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                for dx in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        environment_static[ny, nx] = 1
            obstacle = Obstacle(x, y, is_dynamic=False, size=5, shape='square', color=(255, 255, 255))
            static_obstacles.append(obstacle)
            mark_obstacle(environment_static, x, y, size=5, inflate=SAFE_DISTANCE)

    # 5. 检查起点和终点位置
    if not is_valid_position(start_x, start_y, environment_static, width, height):
        raise ValueError("无效的起点位置")
    if not is_valid_position(goal_x, goal_y, environment_static, width, height):
        raise ValueError("无效的终点位置")

    # 6. 使用FMM算法(类Dijkstra)进行全局路径规划
    start_y_int, start_x_int = int(start_y), int(start_x)
    goal_y_int, goal_x_int = int(goal_y), int(goal_x)
    path = fmm(environment_static, (start_y_int, start_x_int), (goal_y_int, goal_x_int))
    if not path:
        raise ValueError("FMM算法未能找到可行路径")

    # -------------- 打印未平滑路径的统计信息 --------------
    # 1) 节点数
    raw_node_count = len(path)
    # 2) 几何长度
    raw_distance = compute_path_distance(path)
    # 3) 转向点个数(阈值设为15°)
    raw_turns = count_turns_by_angle(path, angle_threshold_degrees=15)

    print(f"未平滑路径节点数: {raw_node_count}")
    print(f"未平滑路径的几何长度(像素): {raw_distance:.2f}")
    print(f"未平滑路径的转向点数(>=15°): {raw_turns}")

    # -------------------- 路径平滑处理 --------------------
    smoothed_path = smooth_path(path, smooth_factor=SMOOTH_FACTOR)

    # -------------- 打印平滑后路径的统计信息 --------------
    smooth_node_count = len(smoothed_path)
    smooth_distance = compute_path_distance(smoothed_path)
    smooth_turns = count_turns_by_angle(smoothed_path, angle_threshold_degrees=15)

    print(f"平滑后路径节点数: {smooth_node_count}")
    print(f"平滑后路径的几何长度(像素): {smooth_distance:.2f}")
    print(f"平滑后路径的转向点数(>=15°): {smooth_turns}")

    # 7. 进入主循环进行可视化
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # 绘制背景
        win.blit(background, (0, 0))

        # 绘制FMM路径（平滑后）
        if len(smoothed_path) > 1:
            path_display = [(px, py) for (py, px) in smoothed_path]
            draw_path(win, path_display, color=(255, 255, 0), width=2)

        # 绘制起点和终点
        draw_star(win, (0, 255, 0), (start_x, start_y), 10)  # 起点为绿色星形
        draw_star(win, (255, 0, 0), (goal_x, goal_y), 10)   # 终点为红色星形

        # 绘制所有静态障碍物
        for obs in static_obstacles:
            obs.draw(win)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

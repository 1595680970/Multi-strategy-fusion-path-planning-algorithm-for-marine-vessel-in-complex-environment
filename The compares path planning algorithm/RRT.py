import pygame
import numpy as np
import math
import random
import matplotlib
matplotlib.use('Agg')  # 若无需GUI可用非交互式后端
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
SAFE_DISTANCE = 10       # 障碍物安全膨胀距离
GOAL_THRESHOLD = 10      # 到达目标判定阈值
SMOOTH_FACTOR = 300      # 路径平滑因子
lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

# ========== 工具函数（路径长度、转向点、平滑）==========
def compute_path_distance(path):
    total_dist = 0.0
    for i in range(len(path) - 1):
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        total_dist += math.hypot(x2 - x1, y2 - y1)
    return total_dist

def count_turns_by_angle(path, angle_threshold_degrees=15):
    if len(path) < 3:
        return 0
    turn_count = 0
    for i in range(1, len(path) - 1):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 < 1e-9 or mag2 < 1e-9:
            continue
        dot_val = v1[0]*v2[0] + v1[1]*v2[1]
        cos_val = dot_val / (mag1 * mag2)
        cos_val = max(-1.0, min(1.0, cos_val))  # 防止浮点误差
        angle_deg = math.degrees(math.acos(cos_val))
        if angle_deg >= angle_threshold_degrees:
            turn_count += 1
    return turn_count

def smooth_path(path, smooth_factor=300):
    if len(path) < 3:
        return path
    y, x = zip(*path)
    try:
        tck, u = splprep([x, y], s=smooth_factor)
        num_points = max(100, len(path))
        u_new = np.linspace(0, 1, num=num_points)
        x_new, y_new = splev(u_new, tck)
        return list(zip(y_new, x_new))
    except Exception:
        return path

# ========== 生成地图并转换为 Pygame.Surface ==========

def generate_map_surface():
    """生成地图，并转为 pygame.Surface + RGB 数组。"""
    dpi = 100
    fig = plt.figure(figsize=(WINDOW_WIDTH / dpi, WINDOW_HEIGHT / dpi), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # 设置经纬度刻度
    xticks = np.arange(lon_min, lon_max + 0.1, 0.2)
    yticks = np.arange(lat_min, lat_max + 0.1, 0.2)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{lon:.1f}°E" for lon in xticks])
    ax.set_yticklabels([f"{lat:.1f}°N" for lat in yticks])

    ax.set_title("Path Planning Using RRT Algorithm", fontsize=16)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # 取像素并关闭图像
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    surface = pygame.image.frombuffer(buf[:, :, :3].tobytes(), (w, h), "RGB")
    return surface, buf[:, :, :3]

def create_environment_from_array(arr):
    """将 RGB 数组转换为环境网格：蓝色=0（可行），其他=1（障碍）。"""
    h, w, _ = arr.shape
    env = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = arr[y, x]
            if b > r and b > g:
                env[y, x] = 0  # 海洋区域
            else:
                env[y, x] = 1  # 陆地或障碍
    return env

def inflate_obstacles(env, distance):
    """
    对环境中的障碍物进行膨胀 distance 个像素，保证与障碍物保持安全距离。
    """
    structure = ndi.generate_binary_structure(2, 1)
    env_copy = env.copy()
    for _ in range(distance):
        env_copy = ndi.binary_dilation(env_copy, structure=structure).astype(env_copy.dtype)
    return env_copy

def latlon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, w, h):
    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h
    return x, y

# ========== 结构化障碍物示例 ==========

class Obstacle:
    """
    简单示例，记录一个静态障碍物的坐标和大小，可在 Pygame 中绘制。
    """
    def __init__(self, x, y, size=5, shape='circle', color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.size = size
        self.shape = shape
        self.color = color

    def draw(self, win):
        if self.shape == 'circle':
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.size)
        elif self.shape == 'square':
            pygame.draw.rect(win, self.color,
                             (int(self.x) - self.size, int(self.y) - self.size,
                              self.size*2, self.size*2))

def mark_obstacle(env, x, y, size=5):
    """
    在环境中将 (x, y) 以及周围 size 范围标记为障碍物。
    """
    h, w = env.shape
    for dy in range(-size, size+1):
        for dx in range(-size, size+1):
            yy = int(y + dy)
            xx = int(x + dx)
            if 0 <= yy < h and 0 <= xx < w:
                env[yy, xx] = 1

# ========== RRT 实现及多次运行统计 ==========

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (y, x)
        self.parent = parent

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_random_point(width, height):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    return (y, x)

def nearest_node(tree, point):
    closest_node = tree[0]
    min_dist = euclidean_distance(closest_node.position, point)
    for node in tree:
        dist = euclidean_distance(node.position, point)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

def steer(from_node, to_point, step_size=10):
    from_y, from_x = from_node.position
    to_y, to_x = to_point
    theta = math.atan2(to_y - from_y, to_x - from_x)
    new_x = from_x + step_size * math.cos(theta)
    new_y = from_y + step_size * math.sin(theta)
    return (int(new_y), int(new_x))

def collision_free(from_pos, to_pos, grid):
    """
    用 Bresenham 直线算法检测两点连线是否有碰撞。
    grid[y, x] == 1 表示障碍，0 表示可行。
    """
    y0, x0 = from_pos
    y1, x1 = to_pos
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    y, x = y0, x0
    n = 1 + dy + dx
    y_inc = 1 if y1 > y0 else -1
    x_inc = 1 if x1 > x0 else -1
    error = dx - dy
    dx <<= 1
    dy <<= 1
    for _ in range(n):
        if grid[y, x] == 1:
            return False
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return True

def rrt(start, goal, grid, max_iterations=5000, step_size=10, goal_sample_rate=0.05):
    tree = [Node(start)]
    for i in range(max_iterations):
        if random.random() < goal_sample_rate:
            rnd_point = goal
        else:
            rnd_point = get_random_point(grid.shape[1], grid.shape[0])

        nearest = nearest_node(tree, rnd_point)
        new_pos = steer(nearest, rnd_point, step_size)

        if not (0 <= new_pos[0] < grid.shape[0] and 0 <= new_pos[1] < grid.shape[1]):
            continue
        if grid[new_pos[0], new_pos[1]] == 1:
            continue
        if not collision_free(nearest.position, new_pos, grid):
            continue

        new_node = Node(new_pos, nearest)
        tree.append(new_node)

        # 到达目标附近
        if euclidean_distance(new_node.position, goal) <= step_size:
            if collision_free(new_node.position, goal, grid):
                goal_node = Node(goal, new_node)
                tree.append(goal_node)
                path = []
                current = goal_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1], i + 1
    return [], max_iterations

def run_rrt_multiple_times(num_runs, start, goal, grid):
    best_path = None
    best_iter = 0
    best_length = float('inf')
    lengths = []

    for _ in range(num_runs):
        path, iters = rrt(start, goal, grid,
                          max_iterations=5000,
                          step_size=10,
                          goal_sample_rate=0.05)
        if path:
            dist = compute_path_distance(path)
            lengths.append(dist)
            if dist < best_length:
                best_length = dist
                best_iter = iters
                best_path = path

    if len(lengths) == 0:
        return [], 0, float('inf'), float('inf')

    avg_length = sum(lengths) / len(lengths)
    return best_path, best_iter, best_length, avg_length

# ========== Pygame 绘制函数 ==========

def draw_star(win, color, position, size):
    x, y = position
    points = []
    for i in range(5):
        angle = i * 2 * math.pi / 5 - math.pi / 2
        outer = (x + size * math.cos(angle), y + size * math.sin(angle))
        inner_angle = angle + math.pi / 5
        inner = (x + size / 2 * math.cos(inner_angle), y + size / 2 * math.sin(inner_angle))
        points.extend([outer, inner])
    pygame.draw.polygon(win, color, points)

# -------------------- 主函数 --------------------
def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Path Planning Using RRT Algorithm (Multiple Runs)")
    clock = pygame.time.Clock()

    # 1. 生成地图背景 & 环境栅格
    background, bg_array = generate_map_surface()
    h, w, _ = bg_array.shape

    env_static = create_environment_from_array(bg_array)

    # 2. 放置一些人工障碍物（示例）
    #   假设我们有几个障碍物坐标 (x, y)，并在环境中标记
    static_obstacle_positions = [
        (580, 397), (294, 159), (554, 344), (435, 194), (216, 117), (395, 453),
        (260, 434), (371, 485), (197, 151), (225, 285), (394, 229), (202, 209),
        (555, 233), (246, 367), (276, 174), (392, 173), (273, 353), (191, 261),
        (568, 280), (468, 181)
    ]
    static_obstacles = []
    for obs_pos in static_obstacle_positions:
        ox, oy = obs_pos
        # 标记障碍物在环境上 (这里 size=5 仅示例，可自行调整)
        mark_obstacle(env_static, ox, oy, size=5)
        # 记录一下，以便后续在pygame上绘制
        obstacle = Obstacle(ox, oy, size=5, shape='square', color=(255, 255, 255))
        static_obstacles.append(obstacle)

    # 3. 对障碍物进行安全膨胀
    env_static = inflate_obstacles(env_static, SAFE_DISTANCE)

    # 4. 设置起点与终点
    fixed_start_lat, fixed_start_lon = 12.2, 123.3
    fixed_goal_lat, fixed_goal_lon = 12.5, 123.8

    sx, sy = latlon_to_pixel(fixed_start_lat, fixed_start_lon,
                             lat_min, lat_max, lon_min, lon_max, w, h)
    gx, gy = latlon_to_pixel(fixed_goal_lat, fixed_goal_lon,
                             lat_min, lat_max, lon_min, lon_max, w, h)

    start = (int(sy), int(sx))  # (y, x)
    goal = (int(gy), int(gx))  # (y, x)

    # -- 额外清理起点/终点周围障碍物 --
    buffer_size = SAFE_DISTANCE + 5
    for dy in range(-buffer_size, buffer_size+1):
        for dx in range(-buffer_size, buffer_size+1):
            ny, nx = start[0] + dy, start[1] + dx
            if 0 <= ny < h and 0 <= nx < w:
                env_static[ny, nx] = 0
    for dy in range(-buffer_size, buffer_size+1):
        for dx in range(-buffer_size, buffer_size+1):
            ny, nx = goal[0] + dy, goal[1] + dx
            if 0 <= ny < h and 0 <= nx < w:
                env_static[ny, nx] = 0


    # 5. 多次运行 RRT
    NUM_RUNS = 100
    best_path, best_iter, best_len, avg_len = run_rrt_multiple_times(NUM_RUNS, start, goal, env_static)

    if not best_path:
        print("多次运行 RRT 均未找到任何可行路径!")
        pygame.quit()
        return

    # 6. 对“最优路径”进行统计 & 平滑
    raw_node_count = len(best_path)
    raw_distance = compute_path_distance(best_path)
    raw_turns = count_turns_by_angle(best_path, angle_threshold_degrees=15)

    smoothed_path = smooth_path(best_path, SMOOTH_FACTOR)
    smooth_node_count = len(smoothed_path)
    smooth_distance = compute_path_distance(smoothed_path)
    smooth_turns = count_turns_by_angle(smoothed_path, angle_threshold_degrees=15)

    # 7. 控制台打印实验信息
    print("========== 多次运行 RRT 统计结果 ==========")
    print(f"运行次数: {NUM_RUNS}")
    print(f"最优路径长度(像素): {best_len:.2f}")
    print(f"平均路径长度(像素): {avg_len:.2f}")

    print("\n========== 最优路径的详细信息 ==========")
    print(f"找到该最优路径时的迭代次数: {best_iter}")
    print(f"未平滑路径节点数: {raw_node_count}")
    print(f"未平滑路径长度(像素): {raw_distance:.2f}")
    print(f"未平滑路径转向点数(>=15°): {raw_turns}")
    print(f"平滑后路径节点数: {smooth_node_count}")
    print(f"平滑后路径长度(像素): {smooth_distance:.2f}")
    print(f"平滑后路径转向点数(>=15°): {smooth_turns}")
    print(f"最优路径长度(与未平滑路径长度应相同): {best_len:.2f}")
    print(f"平均路径长度: {avg_len:.2f}")

    # -------------------- 主循环 (演示路径绘制) --------------------
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # 绘制背景地图
        win.blit(background, (0, 0))

        # 绘制平滑后的最优路径
        if len(smoothed_path) > 1:
            path_xy = [(px, py) for (py, px) in smoothed_path]
            pygame.draw.lines(win, (0, 0, 0), False, path_xy, 2)

        # 绘制起点与终点（五角星标识）
        draw_star(win, (0, 255, 0), (start[1], start[0]), 10)
        draw_star(win, (255, 0, 0), (goal[1], goal[0]), 10)

        # 绘制障碍物
        for obs in static_obstacles:
            obs.draw(win)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

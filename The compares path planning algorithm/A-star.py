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
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
SAFE_DISTANCE = 10       # 安全距离（像素）
OBSTACLE_BUFFER = 15     # 障碍物安全缓冲区
GOAL_THRESHOLD = 10      # 到达目标的阈值（像素）
SMOOTH_FACTOR = 300      # 路径平滑因子

lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

# -------------------- 几何计算辅助函数 --------------------
def compute_path_distance(path):
    """计算一条路径的总几何长度（逐段欧几里得距离之和）。"""
    if len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i+1][1] - path[i][1],
                          path[i+1][0] - path[i][0])
               for i in range(len(path) - 1))


def count_turns_by_angle(path, angle_threshold_degrees=15):
    """统计路径中方向变化大于指定角度阈值的转向点数量。"""
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
        if mag1 < 1e-9 or mag2 < 1e-9:  # 避免零长度向量
            continue

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_val = dot / (mag1 * mag2)
        cos_val = np.clip(cos_val, -1.0, 1.0)  # 数值保护
        angle_degrees = math.degrees(math.acos(cos_val))

        if angle_degrees >= angle_threshold_degrees:
            turn_count += 1
    return turn_count


# -------------------- 辅助函数 --------------------
def generate_map_surface():
    """基于 cartopy 生成地图底图并转换为 pygame.Surface 与 np.array。"""
    dpi = 100
    fig = plt.figure(figsize=(WINDOW_WIDTH / dpi, WINDOW_HEIGHT / dpi), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # 英文标题
    ax.set_title("Path Planning Using A-star Algorithm", fontsize=16)

    # 显示经纬度刻度和标签
    xticks = np.arange(lon_min, lon_max + 0.1, 0.2)
    yticks = np.arange(lat_min, lat_max + 0.1, 0.2)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{lon:.1f}°E" for lon in xticks], fontsize=10)
    ax.set_yticklabels([f"{lat:.1f}°N" for lat in yticks], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    # 只取RGB部分
    surface = pygame.image.frombuffer(buf[:, :, :3].tobytes(), (w, h), "RGB")
    return surface, buf[:, :, :3]


def create_environment_from_array(arr):
    """
    将输入的 RGB 数组转换为环境网格：
    arr[y, x, (R,G,B)], 若像素以蓝色为主则视为海洋(0)，
    否则视为陆地(1)。
    """
    # 向量化处理
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    # b 分量大于 r 和 g 则判为海洋=0，否则陆地=1
    env = np.where((b > r) & (b > g), 0, 1)
    return env


def is_valid_position(x, y, environment, width, height):
    """ 判断 x,y 是否在环境有效范围内、且非陆地。 """
    if not (0 <= x < width and 0 <= y < height):
        return False
    if x < 20 or x > width - 20 or y < 20 or y > height - 20:
        return False
    return (environment[int(y), int(x)] == 0)


def latlon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, w, h):
    """ 将 lat, lon 转换为像素坐标 (x, y)。 """
    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h
    return x, y


def inflate_obstacles(env, distance):
    """使用多次 binary_dilation 对障碍物进行膨胀，以保证安全距离。"""
    structure = ndi.generate_binary_structure(2, 1)  # 4连通结构
    # 直接使用 iterations 参数代替手动循环
    env_dilated = ndi.binary_dilation(env, structure=structure, iterations=distance)
    return env_dilated.astype(env.dtype)


# -------------------- A*算法实现 --------------------
def a_star(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + distance(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


def heuristic(a, b):
    """曼哈顿距离。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(node, grid):
    """ 获取 4 连通邻居。 """
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    max_y, max_x = grid.shape
    for dy, dx in directions:
        ny, nx = node[0] + dy, node[1] + dx
        if 0 <= ny < max_y and 0 <= nx < max_x:
            if grid[ny, nx] == 0:  # 0为可通行
                neighbors.append((ny, nx))
    return neighbors


def distance(a, b):
    """ A*中每一步代价设为1。 """
    return 1


def reconstruct_path(came_from, current):
    """ 从 came_from 映射中回溯路径。 """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


# -------------------- 可视化函数（去除保存与日志） --------------------
def visualize_path(environment, path, start, goal):
    plt.figure(figsize=(8, 6))
    plt.imshow(environment, cmap='Greys', origin='lower')
    if path:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, color='blue', linewidth=2, label='A* Path')
    plt.scatter([start[0]], [start[1]], color='green', s=100, label='Start')
    plt.scatter([goal[0]], [goal[1]], color='red', s=100, label='Goal')
    plt.close()


def visualize_all_obstacles(environment, static_obstacles, start, goal):
    plt.figure(figsize=(8, 6))
    plt.imshow(environment, cmap='Greys', origin='lower')
    for obs in static_obstacles:
        if obs.shape == 'square':
            plt.scatter(obs.x, obs.y, marker='s', color='white', s=100)
        elif obs.shape == 'circle':
            plt.scatter(obs.x, obs.y, marker='o', color='white', s=100)
    plt.scatter([start[0]], [start[1]], color='green', s=100, label='Start')
    plt.scatter([goal[0]], [goal[1]], color='red', s=100, label='Goal')
    plt.close()


# -------------------- 障碍物类定义 --------------------
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
            pygame.draw.rect(
                win,
                self.color,
                (int(self.x) - self.size, int(self.y) - self.size, self.size * 2, self.size * 2)
            )
        elif self.shape == 'triangle':
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            pygame.draw.polygon(win, self.color, points)


def mark_obstacle(env, x, y, size, inflate):
    """ 在 env 中以 (x,y) 为中心标记障碍物并膨胀 inflate 范围。 """
    h, w = env.shape
    for dy in range(-inflate, inflate + 1):
        for dx in range(-inflate, inflate + 1):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < h and 0 <= nx < w:
                env[ny, nx] = 1


def smooth_path(path, smooth_factor=300):
    """ 使用 B 样条对路径进行平滑处理。 """
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


def draw_star(win, color, position, size):
    """ 在 pygame 上绘制一个五角星。 """
    x, y = position
    points = []
    for i in range(5):
        angle = i * 2 * math.pi / 5 - math.pi / 2
        outer = (x + size * math.cos(angle), y + size * math.sin(angle))
        inner_angle = angle + math.pi / 5
        inner = (x + size / 2 * math.cos(inner_angle), y + size / 2 * math.sin(inner_angle))
        points.extend([outer, inner])
    pygame.draw.polygon(win, color, points)


def can_place_obstacle(x, y, obstacles, environment, new_size, buffer_distance=SAFE_DISTANCE):
    """
    检查新障碍物与环境和现有障碍物是否冲突。
    如果该点处是陆地(env==1)，或距离其他障碍物过近，则返回 False。
    """
    if environment[int(y), int(x)] == 1:
        return False
    for obs in obstacles:
        distance_val = math.hypot(x - obs.x, y - obs.y)
        if distance_val < (obs.size + new_size + buffer_distance):
            return False
    return True


def clear_area_around(env, center, radius):
    """ 在 env 中清理 center 附近 radius 范围内的障碍物，使之可通行。 """
    cx, cy = map(int, center)
    h, w = env.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                env[ny, nx] = 0


def main():
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Path Planning Using A-star Algorithm")
    clock = pygame.time.Clock()

    # 1. 环境初始化
    background, bg_array = generate_map_surface()
    height, width, _ = bg_array.shape
    environment_static = create_environment_from_array(bg_array)

    # 2. 膨胀陆地
    environment_static = inflate_obstacles(environment_static, SAFE_DISTANCE)
    environment = environment_static.copy()  # 当前环境

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

    # 清理起点/终点区域
    buffer_size = SAFE_DISTANCE + 5
    clear_area_around(environment_static, start, buffer_size)
    clear_area_around(environment_static, goal, buffer_size)

    # 4. 生成静态障碍物
    static_obstacle_positions = [
        (580, 397), (294, 159), (554, 344), (435, 194), (216, 117), (395, 453),
        (260, 434), (371, 485), (197, 151), (225, 285), (394, 229), (202, 209),
        (555, 233), (246, 367), (276, 174), (392, 173), (273, 353), (191, 261),
        (568, 280), (468, 181)
    ]
    static_obstacles = []
    for (x, y) in static_obstacle_positions:
        if can_place_obstacle(x, y, static_obstacles, environment_static, new_size=5):
            # 将 (x, y) 附近设置为障碍物
            for dyy in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                for dxx in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                    ny, nx = y + dyy, x + dxx
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

    # 6. 使用A*算法
    start_y_int, start_x_int = int(start_y), int(start_x)
    goal_y_int, goal_x_int = int(goal_y), int(goal_x)
    path = a_star((start_y_int, start_x_int), (goal_y_int, goal_x_int), environment_static)
    if not path:
        raise ValueError("A*算法未能找到可行路径")

    # -------------- 未平滑路径统计信息 --------------
    raw_node_count = len(path)
    raw_distance = compute_path_distance(path)
    raw_turns = count_turns_by_angle(path)

    print(f"未平滑路径节点数: {raw_node_count}")
    print(f"未平滑路径的几何长度(像素): {raw_distance:.2f}")
    print(f"未平滑路径的转向点数(>=15°): {raw_turns}")

    # -------------------- 路径平滑处理 --------------------
    smoothed_path = smooth_path(path, smooth_factor=SMOOTH_FACTOR)

    # -------------- 平滑后路径统计信息 --------------
    smooth_node_count = len(smoothed_path)
    smooth_distance = compute_path_distance(smoothed_path)
    smooth_turns = count_turns_by_angle(smoothed_path)

    print(f"平滑后路径节点数: {smooth_node_count}")
    print(f"平滑后路径的几何长度(像素): {smooth_distance:.2f}")
    print(f"平滑后路径的转向点数(>=15°): {smooth_turns}")

    # 7. 合并障碍物
    obstacles = static_obstacles

    # 8. 可视化
    visualize_path(environment_static, smoothed_path, (start_x_int, start_y_int), (goal_x_int, goal_y_int))
    visualize_all_obstacles(environment, obstacles, (start_x_int, start_y_int), (goal_x_int, goal_y_int))

    # -------------------- 主循环 --------------------
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # 绘制背景
        win.blit(background, (0, 0))

        # 绘制平滑后的路径
        if len(smoothed_path) > 1:
            path_xy = [(px, py) for (py, px) in smoothed_path]
            pygame.draw.lines(win, (255, 255, 255), False, path_xy, 2)

        # 绘制起点与终点
        draw_star(win, (0, 255, 0), (start_x, start_y), 10)
        draw_star(win, (255, 0, 0), (goal_x, goal_y), 10)

        # 绘制障碍物
        for obs in obstacles:
            obs.draw(win)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

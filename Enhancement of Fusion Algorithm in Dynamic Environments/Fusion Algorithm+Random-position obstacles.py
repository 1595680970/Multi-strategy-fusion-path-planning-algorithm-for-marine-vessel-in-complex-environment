import pygame
import numpy as np
import math
import random
import heapq
import matplotlib

# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import scipy.ndimage as ndi  # 新增: 引入scipy.ndimage
from scipy.interpolate import splprep, splev  # 新增: 引入样条插值所需的函数

# -------------------- 配置日志 --------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
DT = 0.2  # 时间步长（秒）
MAX_SPEED = 8.0  # 动态障碍物最大速度（像素/秒）
MAX_ACCEL = 0.5  # 最大加速度（像素/秒²），从1.0降为0.5
MAX_DECEL = -2.0  # 最大减速度（像素/秒²），从-1.0降为-2.0
MAX_ANGULAR_VEL = 1.0  # 最大角速度（弧度/秒）
SHIP_MAX_U = 5.0  # 船舶前向最大速度（像素/秒），从10.0降为5.0
SHIP_MAX_V = 5.0  # 船舶横向最大速度（像素/秒），从10.0降为5.0
SAFE_DISTANCE = 10  # 安全距离（像素），从10降为7
PATH_WEIGHT = 2.0  # 路径偏离权重，增加到1.5
TURN_PENALTY = 0.3  # 转向惩罚
GOAL_WEIGHT = 2.0  # 目标距离权重，降低到2.0
OBSTACLE_AVOID_PENALTY = 3000  # 避障惩罚，增加到3000
LOW_SPEED_PENALTY = 400  # 低速惩罚
GOAL_THRESHOLD = 10  # 到达目标的阈值（像素）
OBSTACLE_BUFFER =15 # 障碍物安全缓冲区

lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

# 船体参数（可根据实际情况调整）
m11 = 25.8 # 惯性矩阵元素
m22 = 33.8 # 惯性矩阵元素
d22 = 17.0   # 阻尼系数

# -------------------- 辅助函数 --------------------

def generate_map_surface():
    dpi = 100
    fig = plt.figure(figsize=(WINDOW_WIDTH / dpi, WINDOW_HEIGHT / dpi), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    # 英文标题
    ax.set_title("Path Planning based on FMM and DWA Integration", fontsize=16)

    # 显示经纬度刻度和标签
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
    return surface, buf[:, :, :3]  # 返回RGB部分

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

    参数：
        env (ndarray): 环境矩阵，0表示海洋，1表示障碍物（陆地）。
        distance (int): 需要膨胀的像素距离。

    返回：
        ndarray: 膨胀后的环境矩阵。
    """
    structure = ndi.generate_binary_structure(2, 1)  # 4连通结构
    env = env.copy()
    for _ in range(max(distance - 1, 0)):  # 减少膨胀次数，从原来的distance减小3
        env = ndi.binary_dilation(env, structure=structure).astype(env.dtype)
    return env

def fmm(env, start, goal):
    rows, cols = env.shape
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0
    pq = [(0, start)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    visited = np.zeros((rows, cols), dtype=bool)

    while pq:
        d, (cy, cx) = heapq.heappop(pq)
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        logging.debug(f"FMM搜索中: 当前节点=({cy}, {cx}), 距离={d}")
        if (cy, cx) == goal:
            logging.debug("FMM算法到达目标节点")
            break
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols and env[ny, nx] == 0:
                nd = d + math.sqrt(dy ** 2 + dx ** 2)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, (ny, nx)))
    if dist[goal] == np.inf:
        logging.error("FMM算法未能找到可行路径")
        return []
    path = []
    current = goal
    while current != start:
        path.append(current)
        cy, cx = current
        candidates = [(cy + dy, cx + dx) for dy, dx in directions
                      if 0 <= cy + dy < rows and 0 <= cx + dx < cols]
        current = min(candidates, key=lambda c: dist[c])
    path.append(start)
    path.reverse()
    return path

def nearest_point_on_path(y, x, path):
    min_dist = float('inf')
    closest = path[0]
    for py, px in path:
        d = math.hypot(x - px, y - py)
        if d < min_dist:
            min_dist = d
            closest = (py, px)
    return closest

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

def is_movement_range_clear(obs, environment_static):
    """
    检查动态障碍物的整个运动范围是否清晰（仅海洋区域）
    """
    if obs.movement_type == 'horizontal':
        # 采样运动范围内的位置
        step = max(1, int(obs.speed))  # 确保步长至少为1
        for x in range(int(obs.min_x), int(obs.max_x) + 1, step):
            if environment_static[int(obs.y), x] == 1:
                return False
    elif obs.movement_type == 'vertical':
        step = max(1, int(obs.speed))
        for y in range(int(obs.min_y), int(obs.max_y) + 1, step):
            if environment_static[y, int(obs.x)] == 1:
                return False
    return True

def mark_obstacle(env, x, y, size, inflate):
    """标记障碍物及其膨胀区域"""
    for dy in range(-inflate, inflate + 1):
        for dx in range(-inflate, inflate + 1):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < env.shape[0] and 0 <= nx < env.shape[1]:
                env[ny, nx] = 1

def evaluate_trajectory(ship, a, r, dt, goal, obstacles, path):
    # 根据简化的船体模型进行轨迹评估
    temp_x = ship.x
    temp_y = ship.y
    temp_psi = ship.psi
    temp_u = ship.u
    temp_v = ship.v
    sim_steps = int(2 / dt)
    obs_predictions = []
    for obs in obstacles:
        obs_pred = predict_obstacle_position(obs, dt, sim_steps)
        obs_predictions.append((obs, obs_pred))
    score = 0
    for step in range(sim_steps):
        # 更新状态
        temp_x += (temp_u * math.cos(temp_psi) - temp_v * math.sin(temp_psi)) * dt
        temp_y += (temp_u * math.sin(temp_psi) + temp_v * math.cos(temp_psi)) * dt
        temp_psi += r * dt
        temp_psi = temp_psi % (2 * math.pi)  # 保持psi在0到2π之间
        temp_u += a * dt
        temp_v += (-m11/m22 * temp_u * r - d22/m22 * temp_v) * dt

        # 限制速度
        temp_u = max(min(temp_u, SHIP_MAX_U), -SHIP_MAX_U)
        temp_v = max(min(temp_v, SHIP_MAX_V), -SHIP_MAX_V)

        # 检查边界
        if not (0 <= temp_x < width and 0 <= temp_y < height):
            return float('inf')
        if environment[int(temp_y), int(temp_x)] == 1:
            return float('inf')

        # 检查与动态障碍物的距离
        for obs, opred in obs_predictions:
            if step < len(opred):
                ox, oy = opred[step]
            else:
                ox, oy = opred[-1]
            dist = math.hypot(temp_x - ox, temp_y - oy)
            if dist < SAFE_DISTANCE:
                return float('inf')
            elif dist < SAFE_DISTANCE * 2:
                score += OBSTACLE_AVOID_PENALTY / (dist + 0.1)

        # 计算路径偏离
        if path:
            nearest = nearest_point_on_path(temp_y, temp_x, path)
            dist_to_path = math.hypot(temp_x - nearest[1], temp_y - nearest[0])
            score += PATH_WEIGHT * dist_to_path

    # 目标距离
    dist_to_goal = math.hypot(temp_x - goal[0], temp_y - goal[1])
    score += GOAL_WEIGHT * dist_to_goal
    # 转向惩罚
    score += TURN_PENALTY * abs(r)
    # 低速惩罚
    if abs(temp_u) < SHIP_MAX_U * 0.2:
        score += LOW_SPEED_PENALTY
    return score

def dwa(ship, goal, dt, obstacles, path):
    # 调整DWA以生成加速度和角速度
    a_samples = np.linspace(MAX_DECEL, MAX_ACCEL, 5)  # 从5个样本减少到3个
    r_samples = np.linspace(-MAX_ANGULAR_VEL, MAX_ANGULAR_VEL, 5)  # 从5个样本减少到3个
    best_score = float('inf')
    best_a = 0
    best_r = 0
    for a in a_samples:
        for r in r_samples:
            score = evaluate_trajectory(ship, a, r, dt, goal, obstacles, path)
            if score < best_score:
                best_score = score
                best_a = a
                best_r = r
    if best_score == float('inf'):
        # 无方案时给低速转向方案
        best_a = 0
        best_r = MAX_ANGULAR_VEL
    return best_a, best_r

def predict_obstacle_position(obs, dt, steps):
    pred = []
    ox, oy = obs.x, obs.y
    direction = obs.direction  # 复制当前方向以避免改变原方向
    if obs.is_dynamic:
        for _ in range(steps):
            if obs.movement_type == 'vertical':
                oy += direction * obs.speed * dt
                if oy < obs.min_y or oy > obs.max_y:
                    direction *= -1
                    oy += direction * obs.speed * dt
            elif obs.movement_type == 'horizontal':
                ox += direction * obs.speed * dt
                if ox < obs.min_x or ox > obs.max_x:
                    direction *= -1
                    ox += direction * obs.speed * dt
            pred.append((ox, oy))
    else:
        for _ in range(steps):
            pred.append((ox, oy))
    return pred

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

# -------------------- 类定义 --------------------
class Ship:
    def __init__(self, x, y, psi):
        self.x = x
        self.y = y
        self.psi = psi  # 船舶朝向
        self.u = 0.0  # 前向速度
        self.v = 0.0  # 横向速度
        self.path_taken = []

    def move(self, a, r, dt):
        # 更新朝向
        self.psi += r * dt
        self.psi = self.psi % (2 * math.pi)  # 保持psi在0到2π之间

        # 更新速度
        self.u += a * dt
        self.v += (-m11/m22 * self.u * r - d22/m22 * self.v) * dt

        # 限制速度
        self.u = max(min(self.u, SHIP_MAX_U), -SHIP_MAX_U)
        self.v = max(min(self.v, SHIP_MAX_V), -SHIP_MAX_V)

        # 更新位置
        self.x += (self.u * math.cos(self.psi) - self.v * math.sin(self.psi)) * dt
        self.y += (self.u * math.sin(self.psi) + self.v * math.cos(self.psi)) * dt

        # 记录路径
        self.path_taken.append((self.x, self.y))

    def stop(self):
        self.u = 0.0
        self.v = 0.0

    def draw(self, win):
        ship_length = 15
        ship_width = 10

        # 旋转180度，朝向变为 psi + math.pi
        rotated_psi = self.psi + math.pi

        # 计算前方（旋转后的朝向）
        front_x = self.x + ship_length * math.cos(rotated_psi)
        front_y = self.y + ship_length * math.sin(rotated_psi)

        # 计算后方左右，角度偏移适当修正（旋转后的朝向）
        rear_left_x = self.x + ship_width * math.cos(rotated_psi + math.radians(150))
        rear_left_y = self.y + ship_width * math.sin(rotated_psi + math.radians(150))
        rear_right_x = self.x + ship_width * math.cos(rotated_psi - math.radians(150))
        rear_right_y = self.y + ship_width * math.sin(rotated_psi - math.radians(150))

        # 计算中央后部（旋转后的朝向）
        center_rear_x = self.x - 5 * math.cos(rotated_psi)
        center_rear_y = self.y - 5 * math.sin(rotated_psi)

        # 形成船舶的四个点（旋转后的形状）
        ship_shape = [
            (front_x, front_y),
            (rear_left_x, rear_left_y),
            (center_rear_x, center_rear_y),
            (rear_right_x, rear_right_y)
        ]

        # 绘制船舶的多边形
        pygame.draw.polygon(win, (0, 255, 0), ship_shape)

        # 绘制路径（如果有）
        if len(self.path_taken) > 1:
            pygame.draw.lines(win, (255, 255, 0), False, self.path_taken, 2)

class Obstacle:
    def __init__(self, x, y, is_dynamic=False, size=5, shape='circle', color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.is_dynamic = is_dynamic
        self.size = size
        self.shape = shape
        self.color = color
        if self.is_dynamic:
            self.movement_type = random.choice(['horizontal', 'vertical'])
            self.direction = 1  # 1 或 -1
            self.speed = MAX_SPEED  # 像素/秒
            if self.movement_type == 'horizontal':
                self.min_x = max(x - 50, self.size + OBSTACLE_BUFFER)
                self.max_x = min(x + 50, width - self.size - OBSTACLE_BUFFER)
            else:
                self.min_y = max(y - 50, self.size + OBSTACLE_BUFFER)
                self.max_y = min(y + 50, height - self.size - OBSTACLE_BUFFER)
        else:
            self.movement_type = None
            self.direction = 0
            self.min_x = self.max_x = self.x
            self.min_y = self.max_y = self.y

    def can_move_to(self, new_x, new_y, obstacles):
        # 检查窗口边界
        if new_x < self.size + OBSTACLE_BUFFER or new_x > width - self.size - OBSTACLE_BUFFER:
            return False
        if new_y < self.size + OBSTACLE_BUFFER or new_y > height - self.size - OBSTACLE_BUFFER:
            return False
        # 检查环境（仅海洋区域）
        if environment_static[int(new_y), int(new_x)] == 1:
            return False
        # 检查与其他障碍物的距离
        for o in obstacles:
            if o is not self:
                distance = math.hypot(new_x - o.x, new_y - o.y)
                if distance < (self.size + o.size + SAFE_DISTANCE):
                    return False
        return True

    def move(self, dt, obstacles):
        if not self.is_dynamic:
            return
        old_x, old_y = self.x, self.y
        if self.movement_type == 'horizontal':
            new_x = self.x + self.direction * self.speed * dt
            new_y = self.y
            # 检查运动范围
            if new_x < self.min_x or new_x > self.max_x:
                self.direction *= -1
                new_x = self.x + self.direction * self.speed * dt
                logging.debug(f"动态障碍物水平方向反转: ID={id(self)}, 新方向={self.direction}")
            # 检查是否可以移动到新位置
            if self.can_move_to(new_x, new_y, obstacles):
                self.x = new_x
            else:
                # 反转方向并尝试移动
                self.direction *= -1
                new_x = self.x + self.direction * self.speed * dt
                if self.can_move_to(new_x, new_y, obstacles):
                    self.x = new_x
                logging.debug(f"动态障碍物水平方向因碰撞反转: ID={id(self)}, 新方向={self.direction}")
        elif self.movement_type == 'vertical':
            new_x = self.x
            new_y = self.y + self.direction * self.speed * dt
            # 检查运动范围
            if new_y < self.min_y or new_y > self.max_y:
                self.direction *= -1
                new_y = self.y + self.direction * self.speed * dt
                logging.debug(f"动态障碍物垂直方向反转: ID={id(self)}, 新方向={self.direction}")
            # 检查是否可以移动到新位置
            if self.can_move_to(new_x, new_y, obstacles):
                self.y = new_y
            else:
                # 反转方向并尝试移动
                self.direction *= -1
                new_y = self.y + self.direction * self.speed * dt
                if self.can_move_to(new_x, new_y, obstacles):
                    self.y = new_y
                logging.debug(f"动态障碍物垂直方向因碰撞反转: ID={id(self)}, 新方向={self.direction}")
        # 记录移动
        if (old_x != self.x) or (old_y != self.y):
            logging.debug(f"动态障碍物移动到: ({self.x:.2f}, {self.y:.2f}), 方向: {self.direction}")

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

# -------------------- 路径平滑函数 --------------------

def smooth_path(path, smooth_factor=300):
    """
    使用样条插值对路径进行平滑处理。

    参数：
        path (list of tuples): 原始路径，包含 (y, x) 点。
        smooth_factor (float): 平滑因子，越大平滑程度越高。

    返回：
        list of tuples: 平滑后的路径，包含 (y, x) 点。
    """
    if len(path) < 3:
        return path  # 路径点过少，无法平滑
    y, x = zip(*path)
    # 使用参数化样条插值
    try:
        tck, u = splprep([x, y], s=smooth_factor)
        # 生成平滑后的点数量
        num_points = max(200, len(path))  # 从100增加到200
        u_new = np.linspace(0, 1, num=num_points)
        x_new, y_new = splev(u_new, tck)
        smoothed_path = list(zip(y_new, x_new))
        return smoothed_path
    except Exception as e:
        logging.error(f"路径平滑失败: {e}")
        return path  # 如果平滑失败，返回原始路径

# -------------------- 主程序 --------------------

def main():
    global width, height, environment_static, environment  # 声明为全局变量以便在类中访问
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Path Planning with FMM and DWA")
    clock = pygame.time.Clock()

    # 1. 环境初始化
    background, bg_array = generate_map_surface()
    height, width, _ = bg_array.shape
    environment_static = create_environment_from_array(bg_array)

    # 2. 膨胀陆地以确保安全距离
    environment_static = inflate_obstacles(environment_static, SAFE_DISTANCE)

    environment = environment_static.copy()  # 当前环境，包括动态障碍物

    # 3. 设置起点和终点
    fixed_start_lat, fixed_start_lon = 12.8, 123.3 #12.2, 123.3
    fixed_goal_lat, fixed_goal_lon = 12.2, 123.8 #12.5, 123.8
    start_x, start_y = latlon_to_pixel(fixed_start_lat, fixed_start_lon, lat_min, lat_max, lon_min, lon_max, width, height)
    goal_x, goal_y = latlon_to_pixel(fixed_goal_lat, fixed_goal_lon, lat_min, lat_max, lon_min, lon_max, width, height)
    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # 确保起点和终点周围有足够的空间
    buffer_size = SAFE_DISTANCE + 5
    for dy in range(-buffer_size, buffer_size + 1):
        for dx in range(-buffer_size, buffer_size + 1):
            ny, nx = int(start_y) + dy, int(start_x) + dx
            if 0 <= ny < height and 0 <= nx < width:
                environment_static[ny, nx] = 0  # 清除起点附近的障碍物

    for dy in range(-buffer_size, buffer_size + 1):
        for dx in range(-buffer_size, buffer_size + 1):
            ny, nx = int(goal_y) + dy, int(goal_x) + dx
            if 0 <= ny < height and 0 <= nx < width:
                environment_static[ny, nx] = 0  # 清除终点附近的障碍物

    # 4. 生成静态障碍物
    static_obstacles = []
    num_static_obstacles = 20
    for _ in range(num_static_obstacles):
        for __ in range(100):
            x = random.randint(OBSTACLE_BUFFER + 50, width - OBSTACLE_BUFFER - 50)
            y = random.randint(OBSTACLE_BUFFER + 50, height - OBSTACLE_BUFFER - 50)
            # 避免障碍物靠近起点和终点
            distance_to_start = math.hypot(x - start_x, y - start_y)
            distance_to_goal = math.hypot(x - goal_x, y - goal_y)
            if distance_to_start < 50 or distance_to_goal < 50:
                continue  # 距离起点或终点太近，跳过
            if can_place_obstacle(x, y, static_obstacles, environment_static, new_size=5):
                # 标记缓冲区
                for dy in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                    for dx in range(-OBSTACLE_BUFFER, OBSTACLE_BUFFER + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            environment_static[ny, nx] = 1
                obstacle = Obstacle(x, y, is_dynamic=False, size=5, shape='square', color=(255, 255, 255))
                static_obstacles.append(obstacle)
                mark_obstacle(environment_static, x, y, size=5, inflate=SAFE_DISTANCE)
                logging.debug(f"生成静态障碍物: ({x}, {y})")
                break

    # 5. 检查起点和终点位置
    if not is_valid_position(start_x, start_y, environment_static, width, height):
        raise ValueError("无效的起点位置")
    if not is_valid_position(goal_x, goal_y, environment_static, width, height):
        raise ValueError("无效的终点位置")

    # 6. 使用FMM算法进行全局路径规划
    start_y_int, start_x_int = int(start_y), int(start_x)
    goal_y_int, goal_x_int = int(goal_y), int(goal_x)
    path = fmm(environment_static, (start_y_int, start_x_int), (goal_y_int, goal_x_int))
    if not path:
        raise ValueError("FMM算法未能找到可行路径")

    # 验证路径上的每个点
    for (py, px) in path:
        if environment_static[py, px] == 1:
            raise ValueError("路径中包含障碍物，请检查障碍物和缓冲区设置。")

    # 记录路径信息
    logging.info(f"生成的路径长度: {len(path)}")
    logging.info(f"路径上的前10个点: {path[:10]}")

    # -------------------- 路径平滑处理 --------------------
    smoothed_path = smooth_path(path, smooth_factor=300)  # 调整平滑因子为100
    logging.info(f"平滑后的路径长度: {len(smoothed_path)}")
    logging.info(f"平滑路径上的前10个点: {smoothed_path[:10]}")

    # 7. 生成动态障碍物
    dynamic_obstacles = []
    num_dynamic_obstacles = 10
    for _ in range(num_dynamic_obstacles):
        for __ in range(100):
            x = random.randint(OBSTACLE_BUFFER + 50, width - OBSTACLE_BUFFER - 50)
            y = random.randint(OBSTACLE_BUFFER + 50, height - OBSTACLE_BUFFER - 50)
            # 确保不覆盖起点和终点，并且与静态障碍物及其他动态障碍物保持安全距离
            if (environment_static[int(y), int(x)] == 0 and
                    math.hypot(x - start[0], y - start[1]) > 50 and
                    math.hypot(x - goal[0], y - goal[1]) > 50 and
                    can_place_obstacle(x, y, dynamic_obstacles, environment_static, new_size=7)):
                dyn = Obstacle(x, y, is_dynamic=True, size=7, shape='triangle', color=(255, 0, 0))
                if is_movement_range_clear(dyn, environment_static):
                    dynamic_obstacles.append(dyn)
                    logging.debug(f"生成动态障碍物: ({x}, {y}), 运动类型: {dyn.movement_type}, 方向: {dyn.direction}, 速度: {dyn.speed}")
                    break

    # 合并所有障碍物
    obstacles = static_obstacles + dynamic_obstacles

    # 记录障碍物信息
    logging.info(f"生成的静态障碍物数量: {len(static_obstacles)}")
    logging.info(f"生成的动态障碍物数量: {len(dynamic_obstacles)}")

    # -------------------- 主循环 --------------------
    ship = Ship(start[0], start[1], 0)
    run = True
    stuck_counter = 0
    max_stuck_steps = 200  # 当小船连续200步无法接近目标时，认为被困
    previous_distance = math.hypot(ship.x - goal[0], ship.y - goal[1])

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        dist_to_goal = math.hypot(ship.x - goal[0], ship.y - goal[1])
        if dist_to_goal < GOAL_THRESHOLD:
            # 已到达终点，小船逐步减速并停止
            if ship.u > 0.1 or ship.v > 0.1:
                ship.u *= 0.9  # 每步将速度减小到90%
                ship.v *= 0.9
                a = 0
                r = 0
                ship.move(a, r, DT)
                logging.debug("小船逐步减速接近终点。")
            else:
                # 速度已足够低，停止船舶
                ship.stop()
                a = 0
                r = 0
                ship.move(a, r, DT)
                logging.info("小船已接近终点并停下。")
        else:
            # 正常规划
            if smoothed_path:
                a, r = dwa(ship, (goal[0], goal[1]), DT, obstacles, smoothed_path)
            else:
                # 如果没有路径，尝试重新规划
                path = fmm(environment_static, (int(ship.y), int(ship.x)), (goal_y_int, goal_x_int))
                if path:
                    smoothed_path = smooth_path(path, smooth_factor=100)
                    logging.info("重新规划路径成功。")
                else:
                    logging.warning("重新规划路径失败。")
                a, r = (0, 0)
            ship.move(a, r, DT)

            # 重置环境为静态环境
            environment = environment_static.copy()

            # 更新动态障碍物的位置
            for obs in dynamic_obstacles:
                obs.move(DT, obstacles)
            # 标记所有障碍物的位置
            for obs in obstacles:
                mark_obstacle(environment, obs.x, obs.y, size=obs.size, inflate=SAFE_DISTANCE)

        # 检测小船是否被困
        current_distance = math.hypot(ship.x - goal[0], ship.y - goal[1])
        if current_distance < previous_distance - 1:
            stuck_counter = 0  # 有进展，重置计数器
        else:
            stuck_counter += 1
            logging.debug(f"Stuck Counter: {stuck_counter}")
            if stuck_counter > max_stuck_steps:
                logging.warning("小船被困，尝试重新规划路径或退出仿真。")
                # 尝试重新规划路径
                path = fmm(environment_static, (int(ship.y), int(ship.x)), (goal_y_int, goal_x_int))
                if path:
                    smoothed_path = smooth_path(path, smooth_factor=100)
                    stuck_counter = 0
                    logging.info("重新规划路径成功。")
                else:
                    logging.warning("重新规划路径失败，退出仿真。")
                    run = False
        previous_distance = current_distance

        # 绘制背景
        win.blit(background, (0, 0))

        # 绘制全局路径
        if len(smoothed_path) > 1:
            path_xy = [(px, py) for (py, px) in smoothed_path]
            pygame.draw.lines(win, (255, 255, 255), False, path_xy, 2)

        # 绘制起点和终点
        draw_star(win, (0, 255, 0), start, 10)  # 起点为绿色星形
        draw_star(win, (255, 0, 0), goal, 10)  # 终点为红色星形

        # 绘制小船
        ship.draw(win)

        # 绘制所有障碍物
        for obs in obstacles:
            obs.draw(win)

        # 更新显示
        pygame.display.flip()
        clock.tick(60)  # 控制帧率为60fps

        # 添加日志输出
        logging.debug(f"小船位置: ({ship.x:.2f}, {ship.y:.2f}), 目标距离: {dist_to_goal:.2f}")

    pygame.quit()

if __name__ == "__main__":
    main()

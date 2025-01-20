import pygame
import numpy as np
import math
import random
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
DT = 0.1
MAX_SPEED = 3.0
MAX_TURN = 1.0

SAFE_DISTANCE = 10
TURN_PENALTY = 0.1
GOAL_WEIGHT = 1.5
OBSTACLE_AVOID_PENALTY = 1000
LOW_SPEED_PENALTY = 200
GOAL_THRESHOLD = 10

lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

SHIP_RADIUS = 15  # 船舶的半径，用于碰撞检测

# -------------------- 地图生成 --------------------
def generate_map_surface():
    dpi = 100
    fig = plt.figure(figsize=(WINDOW_WIDTH / dpi, WINDOW_HEIGHT / dpi), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')  # 移除多余的 ')'

    # 标题使用英文
    ax.set_title("Path Planning with DWA Only", fontsize=16)

    ax.set_xticks(np.arange(lon_min, lon_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{lon:.1f}°E" for lon in np.arange(lon_min, lon_max + 0.1, 0.2)], fontsize=10)
    ax.set_yticklabels([f"{lat:.1f}°N" for lat in np.arange(lat_min, lat_max + 0.1, 0.2)], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # 使用 buffer_rgba 获取 RGBA 数据
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb_buf = buf[:, :, :3]  # 仅保留 RGB 通道
    plt.close(fig)
    surface = pygame.image.frombuffer(rgb_buf.tobytes(), (w, h), "RGB")
    return surface, rgb_buf

# -------------------- Pygame初始化 --------------------
pygame.init()
win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
background, bg_array = generate_map_surface()
height, width, _ = bg_array.shape

# -------------------- 环境创建 --------------------
def create_environment_from_array(arr):
    h, w, _ = arr.shape
    env = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            r, g, b = arr[y, x]
            if b > r and b > g:
                env[y, x] = 0  # 水域
            else:
                env[y, x] = 1  # 陆地或障碍物
    return env

# 创建静态环境和动态环境
static_env = create_environment_from_array(bg_array)
dynamic_env = np.zeros_like(static_env)
planning_env = static_env.copy()  # 用于路径规划的环境数组

# -------------------- 有效位置检查 --------------------
def is_valid_position(x, y):
    if not (0 <= x < width and 0 <= y < height):
        return False
    if x < 20 or x > width - 20 or y < 20 or y > height - 20:
        return False
    if static_env[int(y), int(x)] == 1:
        return False
    return True

# -------------------- 坐标转换函数 --------------------
def latlon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, w, h):
    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h
    return x, y

# -------------------- 类定义 --------------------
class Ship:
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.path_taken = []
        self.collision = False  # 碰撞标志

    def move(self, v, w, dt):
        if self.collision:
            return  # 碰撞后停止移动
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.path_taken.append((self.x, self.y))

    def draw(self, win):
        # 使用多边形表示船舶，确保视觉与碰撞检测一致
        ship_shape = [
            (self.x + 15 * math.cos(self.theta), self.y + 15 * math.sin(self.theta)),
            (self.x + 5 * math.cos(self.theta + 2.5), self.y + 5 * math.sin(self.theta + 2.5)),
            (self.x - 5 * math.cos(self.theta), self.y - 5 * math.sin(self.theta)),
            (self.x + 5 * math.cos(self.theta - 2.5), self.y + 5 * math.sin(self.theta - 2.5))
        ]
        pygame.draw.polygon(win, (0, 255, 0), ship_shape)
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
        self.collision = False  # 碰撞标志
        if self.is_dynamic:
            self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        else:
            self.direction = (0, 0)

    def can_move_to(self, nx, ny, obstacles):
        # 检查新位置是否在水域
        if not (0 <= nx < width and 0 <= ny < height):
            return False
        if static_env[int(ny), int(nx)] == 1:
            return False
        # 检查与其他动态障碍物的距离
        for o in obstacles:
            if o is not self and o.is_dynamic and not o.collision:
                if math.hypot(nx - o.x, ny - o.y) < (self.size + o.size + SAFE_DISTANCE):
                    return False
        # 检查 dynamic_env 是否已被占用
        if dynamic_env[int(ny), int(nx)] == 1:
            return False
        return True

    def move(self, dt, obstacles):
        if not self.is_dynamic or self.collision:
            return
        # 计算新位置
        nx = self.x + self.direction[0] * dt * 20
        ny = self.y + self.direction[1] * dt * 20
        if self.can_move_to(nx, ny, obstacles):
            # 清除旧位置标记
            unmark_obstacle(dynamic_env, self.x, self.y, self.size)
            self.x, self.y = nx, ny
            # 标记新位置
            mark_obstacle(dynamic_env, self.x, self.y, self.size)
            # 移除打印语句以减少控制台输出
            # print(f"Dynamic Obstacle moved to ({self.x:.2f}, {self.y:.2f})")
        else:
            # 反转方向
            self.direction = (-self.direction[0], -self.direction[1])
            # 移除打印语句以减少控制台输出
            # print(f"Dynamic Obstacle at ({self.x:.2f}, {self.y:.2f}) reversed direction to {self.direction}")

    def draw(self, win):
        if self.shape == 'circle':
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.size)
        elif self.shape == 'square':
            pygame.draw.rect(win, self.color, (int(self.x) - self.size, int(self.y) - self.size, self.size * 2, self.size * 2))
        elif self.shape == 'triangle':
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            pygame.draw.polygon(win, self.color, points)

# -------------------- 标记障碍物 --------------------
def mark_obstacle(env, x, y, size, inflate=0):
    """
    标记一个圆形障碍物在环境数组中。
    如果需要考虑船舶的尺寸，可以通过 inflate 参数扩展障碍物的标记范围。
    """
    total_size = size + inflate
    for dy in range(-total_size, total_size + 1):
        for dx in range(-total_size, total_size + 1):
            ny = int(y + dy)
            nx = int(x + dx)
            if 0 <= ny < env.shape[0] and 0 <= nx < env.shape[1]:
                if math.sqrt(dx ** 2 + dy ** 2) <= total_size:
                    env[ny, nx] = 1

def unmark_obstacle(env, x, y, size, inflate=0):
    """
    清除一个圆形障碍物在环境数组中的标记。
    """
    total_size = size + inflate
    for dy in range(-total_size, total_size + 1):
        for dx in range(-total_size, total_size + 1):
            ny = int(y + dy)
            nx = int(x + dx)
            if 0 <= ny < env.shape[0] and 0 <= nx < env.shape[1]:
                if math.sqrt(dx ** 2 + dy ** 2) <= total_size:
                    env[ny, nx] = 0

# -------------------- 生成起点和终点 --------------------
fixed_start_lat, fixed_start_lon = 12.2, 123.3
fixed_goal_lat, fixed_goal_lon = 12.5, 123.8
start_x, start_y = latlon_to_pixel(fixed_start_lat, fixed_start_lon, lat_min, lat_max, lon_min, lon_max, width, height)
goal_x, goal_y = latlon_to_pixel(fixed_goal_lat, fixed_goal_lon, lat_min, lat_max, lon_min, lon_max, width, height)
start = (start_x, start_y)
goal = (goal_x, goal_y)

# 检查起点和终点的有效性
if not is_valid_position(*start):
    raise ValueError("Invalid start position")
if not is_valid_position(*goal):
    raise ValueError("Invalid goal position")

# -------------------- 放置静态障碍物 --------------------
num_static_obstacles = 20
static_obstacles = []
for _ in range(num_static_obstacles):
    for __ in range(100):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        # 确保不覆盖起点和终点，并且与起点、终点保持一定距离
        if (static_env[y, x] == 0 and
                math.hypot(x - start[0], y - start[1]) > 50 and
                math.hypot(x - goal[0], y - goal[1]) > 50):
            size = 5  # 静态障碍物大小一致
            static_obstacles.append(Obstacle(x, y, is_dynamic=False, size=size, shape='square', color=(255, 255, 255)))
            mark_obstacle(static_env, x, y, size=size, inflate=0)  # 标记静态障碍物
            mark_obstacle(planning_env, x, y, size=size, inflate=SHIP_RADIUS)  # 标记规划时的障碍物
            break

# -------------------- 放置动态障碍物 --------------------
num_dynamic_obstacles = 10
dynamic_obstacles = []
for _ in range(num_dynamic_obstacles):
    for __ in range(100):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        # 确保不覆盖起点和终点，并且与静态障碍物及其他动态障碍物保持安全距离
        if (static_env[y, x] == 0 and
                math.hypot(x - start[0], y - start[1]) > 50 and
                math.hypot(x - goal[0], y - goal[1]) > 50 and
                all(math.hypot(x - o.x, y - o.y) > (o.size + SAFE_DISTANCE) for o in dynamic_obstacles)):
            size = 7  # 动态障碍物大小一致
            dyn = Obstacle(x, y, is_dynamic=True, size=size, shape='triangle', color=(255, 0, 0))
            dynamic_obstacles.append(dyn)
            mark_obstacle(dynamic_env, x, y, size=size, inflate=0)  # 标记动态障碍物
            break

obstacles = static_obstacles + dynamic_obstacles

# -------------------- 预测障碍物位置 --------------------
def predict_obstacle_position(obs, dt, steps):
    pred = []
    ox, oy = obs.x, obs.y
    if obs.is_dynamic and not obs.collision:
        for i in range(steps):
            nx = ox + obs.direction[0] * dt * 20
            ny = oy + obs.direction[1] * dt * 20
            # 检查新位置是否合法
            if not (0 <= nx < width and 0 <= ny < height):
                break
            if static_env[int(ny), int(nx)] == 1:
                break
            # 检查与其他障碍物的距离
            collision = False
            for o in obstacles:
                if o is not obs and o.is_dynamic and not o.collision:
                    if math.hypot(nx - o.x, ny - o.y) < (obs.size + o.size + SAFE_DISTANCE):
                        collision = True
                        break
            if collision:
                break
            ox, oy = nx, ny
            pred.append((ox, oy))
        if not pred:
            # 如果预测为空，添加当前位置信息以避免IndexError
            pred.append((ox, oy))
    else:
        for i in range(steps):
            pred.append((ox, oy))
    return pred

# -------------------- 评估轨迹 --------------------
def evaluate_trajectory(ship, v, w, dt, goal, obstacles):
    # 无全局路径，仅以距离目标和避障为准则
    sim_steps = int(2 / dt)
    obs_predictions = []
    for obs in obstacles:
        obs_pred = predict_obstacle_position(obs, dt, sim_steps)
        obs_predictions.append((obs, obs_pred))
    x, y, theta = ship.x, ship.y, ship.theta
    score = 0
    for step in range(sim_steps):
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
        if not (0 <= x < width and 0 <= y < height):
            return float('inf')
        if static_env[int(y), int(x)] == 1:
            return float('inf')
        for obs, opred in obs_predictions:
            if step < len(opred):
                ox, oy = opred[step]
            else:
                ox, oy = opred[-1]
            dist = math.hypot(x - ox, y - oy)
            if dist < SAFE_DISTANCE:
                return float('inf')
            elif dist < SAFE_DISTANCE * 2:
                score += OBSTACLE_AVOID_PENALTY / (dist + 0.1)
    # 没有path，只考虑靠近goal
    dist_to_goal = math.hypot(x - goal[0], y - goal[1])
    score += GOAL_WEIGHT * dist_to_goal
    score += TURN_PENALTY * abs(w)
    if v < MAX_SPEED * 0.2:
        score += LOW_SPEED_PENALTY
    return score

# -------------------- DWA 算法 --------------------
def dwa(ship, goal, dt, obstacles):
    v_samples = np.linspace(0, MAX_SPEED, 5)
    w_samples = np.linspace(-MAX_TURN, MAX_TURN, 5)
    best_score = float('inf')
    best_v = 0
    best_w = 0
    for v in v_samples:
        for w in w_samples:
            score = evaluate_trajectory(ship, v, w, dt, goal, obstacles)
            if score < best_score:
                best_score = score
                best_v = v
                best_w = w
    if best_score == float('inf'):
        best_v = 0
        best_w = MAX_TURN
    return best_v, best_w

# -------------------- 绘制星形标记 --------------------
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

# -------------------- 碰撞检测 --------------------
def check_collision(ship, obstacles):
    for obs in obstacles:
        if obs.collision:
            continue  # 忽略已经碰撞停止的障碍物
        dist = math.hypot(ship.x - obs.x, ship.y - obs.y)
        # 调整碰撞阈值以匹配视觉表示
        collision_threshold = SHIP_RADIUS + obs.size - 2  # 减少一些距离
        if dist < collision_threshold:
            return True
    return False

# -------------------- 主循环 --------------------
ship = Ship(start[0], start[1], 0)
run = True
collision_detected = False  # 初始化碰撞标志

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    dist_to_goal = math.hypot(ship.x - goal[0], ship.y - goal[1])
    if dist_to_goal < GOAL_THRESHOLD:
        # 到达终点停船不关窗
        ship.move(0, 0, DT)
    else:
        v, w = dwa(ship, (goal[0], goal[1]), DT, obstacles)
        ship.move(v, w, DT)

        for obs in dynamic_obstacles:
            # 清除旧位置标记
            unmark_obstacle(dynamic_env, obs.x, obs.y, obs.size)
            # 移动障碍物
            obs.move(DT, obstacles)
            # 标记新位置
            mark_obstacle(dynamic_env, obs.x, obs.y, obs.size)

    # 检查是否与障碍物碰撞
    if not collision_detected and check_collision(ship, obstacles):
        print("Collision detected!")
        collision_detected = True
        ship.collision = True  # 停止移动

    # 绘制
    win.blit(background, (0, 0))
    draw_star(win, (0, 255, 0), start, 10)  # 起点
    draw_star(win, (255, 0, 0), goal, 10)  # 终点
    ship.draw(win)
    for obs in obstacles:
        obs.draw(win)
    if collision_detected:
        # 显示碰撞信息
        font = pygame.font.SysFont(None, 36)
        text = font.render("Collision Detected!", True, (255, 0, 0))
        win.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    clock.tick(10)  # 可以适当调整帧率

pygame.quit()

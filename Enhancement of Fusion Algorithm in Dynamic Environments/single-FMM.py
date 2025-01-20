import pygame
import numpy as np
import math
import random
import heapq
import matplotlib

# 使用 Agg 后端以避免显示问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------- 参数设置 --------------------
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
DT = 0.1
MAX_SPEED = 3.0  # 降低速度以提高碰撞检测准确性
TURN_SPEED = 0.05
GOAL_THRESHOLD = 10

lat_min, lat_max = 12, 13
lon_min, lon_max = 123, 124

SHIP_RADIUS = 15  # 船舶的半径，用于路径规划
SAFE_DISTANCE = 10  # 用于动态障碍物之间的安全距离

# -------------------- 地图生成 --------------------
def generate_map_surface():
    dpi = 100
    fig = plt.figure(figsize=(WINDOW_WIDTH / dpi, WINDOW_HEIGHT / dpi), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAND, facecolor='bisque')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.set_title("Path Planning with FMM Only", fontsize=16)
    ax.set_xticks(np.arange(lon_min, lon_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max + 0.1, 0.2), crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{lon:.1f}°E" for lon in np.arange(lon_min, lon_max + 0.1, 0.2)], fontsize=10)
    ax.set_yticklabels([f"{lat:.1f}°N" for lat in np.arange(lat_min, lat_max + 0.1, 0.2)], fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # 使用 buffer_rgba 获取 RGBA 数据
    rgba_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb_buf = rgba_buf[:, :, :3]  # 只保留 RGB 通道
    plt.close(fig)
    surface = pygame.image.frombuffer(rgb_buf.tobytes(), (w, h), "RGB")
    return surface, rgb_buf  # 返回 RGB 数据

# -------------------- Pygame初始化 --------------------
pygame.init()
win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
background, rgb_array = generate_map_surface()  # 使用 rgb_array
height, width, _ = rgb_array.shape

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
static_env = create_environment_from_array(rgb_array)
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

    def move_towards(self, target_x, target_y, dt):
        if self.collision:
            return  # 碰撞后停止移动
        dx = target_x - self.x
        dy = target_y - self.y
        desired_angle = math.atan2(dy, dx)
        angle_diff = (desired_angle - self.theta + math.pi) % (2 * math.pi) - math.pi
        if angle_diff > 0:
            self.theta += min(angle_diff, TURN_SPEED)
        else:
            self.theta += max(angle_diff, -TURN_SPEED)

        self.x += MAX_SPEED * math.cos(self.theta) * dt
        self.y += MAX_SPEED * math.sin(self.theta) * dt
        self.path_taken.append((self.x, self.y))

    def draw(self, win):
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
        self.size = size  # 保持障碍物大小一致
        self.shape = shape
        self.color = color
        self.motion_pattern = None  # 后续设定
        self.collision = False  # 碰撞标志

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

    def can_move(self, dt, static_env, dynamic_env, obs):
        # 根据 motion_pattern 计算下个位置
        if not self.is_dynamic or not self.motion_pattern or self.collision:
            return self.x, self.y
        m = self.motion_pattern
        speed = m["speed"]
        direction = m.get("direction", 1)
        if m["type"] == "horizontal":
            nx = self.x + speed * dt * direction
            ny = self.y
            if nx < m["range"][0] or nx > m["range"][1]:
                m["direction"] = -direction
                nx = self.x + speed * dt * m["direction"]
        elif m["type"] == "vertical":
            ny = self.y + speed * dt * direction
            nx = self.x
            if ny < m["range"][0] or ny > m["range"][1]:
                m["direction"] = -direction
                ny = self.y + speed * dt * m["direction"]
        else:
            return self.x, self.y

        # 确保不在陆地上
        if not (0 <= nx < width and 0 <= ny < height):
            m["direction"] = -m["direction"]
            return self.x, self.y
        if static_env[int(ny), int(nx)] == 1:
            # 碰到陆地，反向
            m["direction"] = -m["direction"]
            return self.x, self.y
        # 检查是否与其他动态障碍物碰撞
        for o in obs:
            if o is not self and o.is_dynamic and not o.collision:
                if math.hypot(nx - o.x, ny - o.y) < (self.size + o.size + SAFE_DISTANCE):
                    return self.x, self.y
        return nx, ny

    def move(self, dt, static_env, dynamic_env, obs):
        if self.is_dynamic and self.motion_pattern and not self.collision:
            nx, ny = self.can_move(dt, static_env, dynamic_env, obs)
            # 检查是否移动
            if nx == self.x and ny == self.y:
                # 未能移动，可能碰撞，尝试反向
                self.motion_pattern["direction"] *= -1
                nx, ny = self.can_move(dt, static_env, dynamic_env, obs)
                if nx != self.x or ny != self.y:
                    self.x, self.y = nx, ny
            else:
                self.x, self.y = nx, ny

# -------------------- 路径规划 (FMM) --------------------
def fmm(env, start, goal):
    rows, cols = env.shape
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0
    pq = [(0, start)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pq:
        d, (cy, cx) = heapq.heappop(pq)
        if (cy, cx) == goal:
            break
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols and env[ny, nx] == 0:
                nd = d + math.sqrt(dy ** 2 + dx ** 2)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, (ny, nx)))
    if dist[goal] == np.inf:
        return []
    path = []
    current = goal
    while current != start:
        path.append(current)
        cy, cx = current
        candidates = [(cy + dy, cx + dx) for dy, dx in dirs if 0 <= cy + dy < rows and 0 <= cx + dx < cols]
        current = min(candidates, key=lambda c: dist[c])
    path.append(start)
    path.reverse()
    return path

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
            # 随机决定是水平或垂直运动
            if random.choice([True, False]):
                # 水平运动
                motion_pattern = {"type": "horizontal", "range": [max(x - 30, 0), min(x + 30, width - 1)],
                                  "speed": 15, "direction": 1}
            else:
                # 垂直运动
                motion_pattern = {"type": "vertical", "range": [max(y - 30, 0), min(y + 30, height - 1)],
                                  "speed": 15, "direction": 1}
            dyn.motion_pattern = motion_pattern
            dynamic_obstacles.append(dyn)
            mark_obstacle(dynamic_env, x, y, size=size, inflate=0)  # 标记动态障碍物
            break

obstacles = static_obstacles + dynamic_obstacles

# -------------------- 路径规划 --------------------
path = fmm(planning_env, (int(start_y), int(start_x)), (int(goal_y), int(goal_x)))
if not path:
    raise ValueError("No feasible path")

# 将路径从（y, x）转换为（x, y）
path_xy = [(px, py) for (py, px) in path]

# -------------------- 可视化路径（可选） --------------------
def draw_path(win, path, color=(255, 255, 255), width=2):
    if len(path) > 1:
        pygame.draw.lines(win, color, False, path, width)

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

# -------------------- 船舶类（仅FMM） --------------------
class OnlyFMMShip(Ship):
    def __init__(self, x, y, path):
        super().__init__(x, y, 0)
        self.path = path
        self.current_index = 0
        if len(self.path) > 1:
            target_x, target_y = self.path[1]
            self.theta = math.atan2(target_y - self.y, target_x - self.x)

    def update(self, dt):
        if self.collision:
            return  # 碰撞后停止移动
        dist_to_goal = math.hypot(self.x - goal[0], self.y - goal[1])
        if dist_to_goal < GOAL_THRESHOLD:
            self.move_towards(self.x, self.y, dt)  # 停止移动
        else:
            if self.current_index < len(self.path) - 1:
                target_x, target_y = self.path[self.current_index + 1]
                dist = math.hypot(target_x - self.x, target_y - self.y)
                if dist < 5:
                    self.current_index += 1
                    if self.current_index < len(self.path) - 1:
                        target_x, target_y = self.path[self.current_index + 1]
                self.move_towards(target_x, target_y, dt)
            else:
                self.move_towards(self.x, self.y, dt)

# -------------------- 动态障碍物移动 --------------------
def move_dynamic_obstacles(dt):
    for obs in dynamic_obstacles:
        if obs.collision:
            continue  # 已碰撞的障碍物不再移动
        # 清除旧位置标记
        unmark_obstacle(dynamic_env, obs.x, obs.y, size=obs.size, inflate=0)
        # 移动障碍物
        obs.move(dt, static_env, dynamic_env, dynamic_obstacles)
        # 检查是否与静态障碍物或其他动态障碍物相撞
        if static_env[int(obs.y), int(obs.x)] == 1:
            # 碰撞到陆地
            print(f"Dynamic obstacle at ({obs.x}, {obs.y}) collided with land.")
            obs.collision = True
            obs.motion_pattern = None
            continue
        if dynamic_env[int(obs.y), int(obs.x)] == 1:
            # 碰撞到其他动态障碍物
            print(f"Dynamic obstacle at ({obs.x}, {obs.y}) collided with another dynamic obstacle.")
            obs.collision = True
            obs.motion_pattern = None
            continue
        # 重新标记新位置
        mark_obstacle(dynamic_env, obs.x, obs.y, size=obs.size, inflate=0)

# -------------------- 碰撞检测 --------------------
def check_collision(ship, dynamic_obstacles):
    for obs in dynamic_obstacles:
        if obs.collision:
            continue  # 忽略已经碰撞停止的障碍物
        dist = math.hypot(ship.x - obs.x, ship.y - obs.y)
        # 调整碰撞阈值以匹配视觉表示
        collision_threshold = SHIP_RADIUS + obs.size - 2  # 减少一些距离
        if dist < collision_threshold:
            return True
    return False

# -------------------- 主循环 --------------------
# 初始化船舶
ship = OnlyFMMShip(start[0], start[1], path_xy)

collision_detected = False

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    ship.update(DT)
    move_dynamic_obstacles(DT)

    # 检查是否与动态障碍物碰撞
    if not collision_detected and check_collision(ship, dynamic_obstacles):
        print("Collision detected!")
        collision_detected = True
        ship.collision = True  # 停止移动

    win.blit(background, (0, 0))
    # 绘制FMM路径（可选，调试用）
    # draw_path(win, path_xy, color=(255,255,255), width=2)
    # 不绘制FMM路径线，仅绘制起点、终点、障碍物和小船
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
    clock.tick(10)

pygame.quit()



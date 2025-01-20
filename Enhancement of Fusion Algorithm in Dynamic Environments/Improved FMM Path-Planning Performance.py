import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cFeature
from matplotlib.patches import Ellipse
import heapq
import logging

import shapely.geometry as geom
from shapely.ops import unary_union
from shapely.prepared import prep
from collections import OrderedDict
from matplotlib.lines import Line2D

# 使用 Matplotlib 内置的 'classic' 样式
plt.style.use('classic')
print("使用 'classic' 样式。")

# 模拟动态障碍物和静态障碍物的经纬度
dynamic_obstacles = [
    (12.01535, 123.00842),  # 动态障碍物 1
    (11.99912, 121.00892),  # 动态障碍物 2
    (11.33315, 120.5530),    # 动态障碍物 3
    (13.6163, 120.99792)     # 动态障碍物 4
]

static_obstacles = [
    # 静态障碍物列表
    (12.77668, 120.6189),  # 静态障碍物 1
    (11.5824, 120.50353),  # 静态障碍物 2
    (12.88988, 122.6239),  # 静态障碍物 3
    (11.25725, 120.48157),  # 静态障碍物 4
    (12.0586, 120.4596),  # 静态障碍物 5
    (11.48488, 121.05835),  # 静态障碍物 6
    (13.35827, 120.00915),  # 静态障碍物 7
    (13.13765, 121.81092),  # 静态障碍物 9
    (13.36365, 119.92127),  # 静态障碍物 10
    (13.41743, 119.98168),  # 静态障碍物 11
    (13.37440, 119.78943),  # 静态障碍物 12
    (12.99763, 120.16845),  # 静态障碍物 13
    (13.60018, 121.78345),  # 静态障碍物 14
    (11.95045, 120.44312),  # 静态障碍物 15
    (12.06400, 121.30005),  # 静态障碍物 16
    (11.02942, 120.49805),  # 静态障碍物 17
    (11.92340, 120.08057),  # 静态障碍物 18
    (10.75795, 120.54748),  # 静态障碍物 19
    (11.02398, 120.50903),  # 静态障碍物 20
    (10.91543, 120.1355),   # 静态障碍物 21
    (10.32853, 120.40467),  # 静态障碍物 22
    (10.57865, 120.531),    # 静态障碍物 23
    (12.53395, 120.3772),   # 静态障碍物 24
    (10.96428, 120.50903),  # 静态障碍物 25
    (10.93715, 120.15198),  # 静态障碍物 26
    (11.83143, 120.48157),  # 静态障碍物 27
    (11.35483, 121.8274),   # 静态障碍物 28
    (12.72277, 120.22338),  # 静态障碍物 29
    (12.98687, 120.19592),  # 静态障碍物 30
    (12.88448, 118.57543),  # 静态障碍物 31
    (12.71737, 118.70728),  # 静态障碍物 32
    (10.76338, 122.79418),  # 静态障碍物 33
    (13.36903, 118.91602),  # 静态障碍物 34
    (12.12888, 124.40918),  # 静态障碍物 35
    (11.28977, 123.3435)    # 静态障碍物 36
]

# 定义代价函数（基于威胁等级和障碍物）
def calculate_cost_vectorized(lats, lons, dynamic_obstacles, static_obstacles, land_prepared):
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cost_field = np.ones(lon_grid.shape)  # 初始化海洋区域代价值为1

    # 陆地区域设为无穷大
    points = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T
    mask_land = np.array([land_prepared.contains(geom.Point(lon, lat)) for lon, lat in points]).reshape(lon_grid.shape)
    cost_field[mask_land] = np.inf

    # 处理静态障碍物
    for solat, solon in static_obstacles:
        sodist = np.sqrt((lat_grid - solat) ** 2 + (lon_grid - solon) ** 2)
        mask_inf = sodist < 0.05
        mask_high = (sodist >= 0.05) & (sodist < 0.1)
        cost_field[mask_inf] = np.inf
        cost_field[mask_high] += 500.0 * (0.1 - sodist[mask_high]) / 0.05

    # 处理动态障碍物，增加多个高斯层次以形成同心圆效果
    for dlat, dlon in dynamic_obstacles:
        ddist = np.sqrt((lat_grid - dlat) ** 2 + (lon_grid - dlon) ** 2)
        mask = ddist < 1.0
        # 增加多个高斯层次
        cost_field[mask] += 100.0 * np.exp(-4.0 * ddist[mask])  # 第一层
        cost_field[mask] += 50.0 * np.exp(-8.0 * ddist[mask])   # 第二层
        cost_field[mask] += 25.0 * np.exp(-12.0 * ddist[mask])  # 第三层

    return cost_field

def initialize_cost_field_vectorized(lat_range, lon_range, resolution, dynamic_obstacles, static_obstacles,
                                     land_prepared):
    lats = np.linspace(lat_range[0], lat_range[1], resolution)
    lons = np.linspace(lon_range[0], lon_range[1], resolution)
    cost_field = calculate_cost_vectorized(lats, lons, dynamic_obstacles, static_obstacles, land_prepared)
    return lats, lons, cost_field

# 加载陆地多边形数据
def load_land_polygons(scale='10m'):
    # 使用 NaturalEarth 数据集加载陆地特征
    land_feature = cFeature.NaturalEarthFeature('physical', 'land', scale)
    polygons = [geom.shape(geom_data) for geom_data in land_feature.geometries()]
    land_union = unary_union(polygons)
    prepared_land = prep(land_union)
    return prepared_land

# 快速扩展方法 (FMM) 实现
def fast_marching_method(lats, lons, cost_field, start_coord, goal_coord):
    resolution = len(lats)
    times = np.full((resolution, resolution), np.inf)
    visited = np.zeros((resolution, resolution), dtype=bool)

    start_lat, start_lon = start_coord
    goal_lat, goal_lon = goal_coord

    start_i = np.argmin(np.abs(lats - start_lat))
    start_j = np.argmin(np.abs(lons - start_lon))
    goal_i = np.argmin(np.abs(lats - goal_lat))
    goal_j = np.argmin(np.abs(lons - goal_lon))

    times[start_i, start_j] = 0.0
    heap = []
    heapq.heappush(heap, (0.0, (start_i, start_j)))

    while heap:
        current_time, (i, j) = heapq.heappop(heap)
        if visited[i, j]:
            continue
        visited[i, j] = True

        if (i, j) == (goal_i, goal_j):
            break

        # 8邻域
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < resolution and 0 <= nj < resolution and not visited[ni, nj]:
                # 计算距离
                if di != 0 and dj != 0:
                    dist = np.sqrt(2)
                else:
                    dist = 1.0
                new_time = current_time + cost_field[ni, nj] * dist
                if new_time < times[ni, nj]:
                    times[ni, nj] = new_time  # 正确的更新
                    heapq.heappush(heap, (new_time, (ni, nj)))

    return times, (goal_i, goal_j), (start_i, start_j)

# 路径重建
def reconstruct_path(times, start_idx, goal_idx):
    path = []
    current = goal_idx
    path.append(current)
    while current != start_idx:
        i, j = current
        # 8邻域
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        min_time = np.inf
        next_node = current
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < times.shape[0] and 0 <= nj < times.shape[1]:
                if times[ni, nj] < min_time:
                    min_time = times[ni, nj]
                    next_node = (ni, nj)
        if next_node == current:
            print("无法找到路径")
            break
        path.append(next_node)
        current = next_node
    return path[::-1]

# 定义自定义图例
def create_custom_legend(ax):
    legend_elements = [
        Line2D([0], [0], marker='x', color='r', label='Dynamic Obstacle', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='o', color='b', label='Static Obstacle', markersize=5, linestyle='None'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Planned Path'),
        Line2D([0], [0], marker='D', color='green', label='Start', markersize=8, linestyle='None'),
        Line2D([0], [0], marker='*', color='gold', label='Goal', markersize=12, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=False)

# 可视化路径
def plot_path(lats, lons, cost_field, path, dynamic_obstacles, static_obstacles, start_coord, goal_coord,
              smoothed_path=None):
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118, 126, 10, 16])

    # 绘制海洋区域为淡蓝色
    ax.add_feature(cFeature.OCEAN, facecolor='lightblue')
    # 绘制陆地背景
    ax.add_feature(cFeature.LAND, facecolor='lightgray')
    ax.add_feature(cFeature.COASTLINE)
    ax.add_feature(cFeature.BORDERS, linestyle=':')

    # 绘制代价场
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # 避免在代价场中出现无穷大，进行对数归一化
    cost_safe = np.where(cost_field == np.inf, np.nan, cost_field)
    # 使用对数归一化提高低代价区域的对比度
    vmin = np.nanmin(cost_safe[cost_safe > 0]) if np.any(cost_safe > 0) else 1
    vmax = np.nanmax(cost_safe)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    contour = ax.contourf(lon_grid, lat_grid, cost_safe, levels=50, norm=norm, transform=ccrs.PlateCarree(),
                          cmap='viridis', alpha=0.6)  # 使用 'viridis' 配色方案，降低透明度

    # 添加颜色条
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('Cost')

    # 绘制路径
    if len(path) > 0:
        path_lons = [lons[j] for i, j in path]
        path_lats = [lats[i] for i, j in path]
        ax.plot(path_lons, path_lats, color='black', linewidth=2, linestyle='--', label='Planned Path')

    # 绘制起点和终点
    ax.plot(start_coord[1], start_coord[0], marker='D', color='green', markersize=8, label='Start')
    ax.plot(goal_coord[1], goal_coord[0], marker='*', color='gold', markersize=12, label='Goal')

    # 绘制动态障碍物的位置和活动范围
    for idx, (lat, lon) in enumerate(dynamic_obstacles):
        ax.plot(lon, lat, 'rx', markersize=8)
        # 绘制动态障碍物的活动范围，颜色根据距离变化
        for radius, alpha in zip([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]):
            ax.add_patch(
                Ellipse(
                    (lon, lat), width=radius * 2, height=radius * 2,
                    edgecolor='red', facecolor='none', linestyle='--', alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            )

    # 绘制静态障碍物的位置和缓冲区
    for idx, (lat, lon) in enumerate(static_obstacles):
        ax.plot(lon, lat, 'bo', markersize=5)
        # 绘制静态障碍物的缓冲区，颜色根据距离变化
        for radius, alpha in zip([0.05, 0.1], [0.3, 0.2]):
            ax.add_patch(
                Ellipse(
                    (lon, lat), width=radius * 2, height=radius * 2,
                    edgecolor='blue', facecolor='none', linestyle='--', alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            )

    # 设置标题和轴标签
    ax.set_title("Path Planning Using the Improved FMM Algorithm", fontsize=16)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    # 设置经纬度刻度
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xticks(np.arange(118, 127, 1))
    ax.set_yticks(np.arange(10, 17, 1))
    ax.set_xticklabels([f"{int(i)}°E" for i in np.arange(118, 127, 1)], fontsize=10)
    ax.set_yticklabels([f"{int(i)}°N" for i in np.arange(10, 17, 1)], fontsize=10)

    # 添加自定义图例
    create_custom_legend(ax)

    plt.show()

# 代价场可视化
def plot_cost_field(lats, lons, cost_field, dynamic_obstacles, static_obstacles):
    print("plot_cost_field 被调用")
    print(f"lats shape: {lats.shape}, lons shape: {lons.shape}, cost_field shape: {cost_field.shape}")
    print(f"代价场最小值: {np.nanmin(cost_field)}, 最大值: {np.nanmax(cost_field)}")

    # 创建新的图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118, 126, 10, 16])

    # 绘制海洋区域为淡蓝色
    ax.add_feature(cFeature.OCEAN, facecolor='lightblue')
    # 绘制陆地背景
    ax.add_feature(cFeature.LAND, facecolor='lightgray')
    ax.add_feature(cFeature.COASTLINE)
    ax.add_feature(cFeature.BORDERS, linestyle=':')

    # 绘制代价场
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # 避免在代价场中出现无穷大，进行对数归一化
    cost_safe = np.where(cost_field == np.inf, np.nan, cost_field)
    # 使用对数归一化提高低代价区域的对比度
    vmin = np.nanmin(cost_safe[cost_safe > 0]) if np.any(cost_safe > 0) else 1
    vmax = np.nanmax(cost_safe)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    contour = ax.contourf(lon_grid, lat_grid, cost_safe, levels=50, norm=norm, transform=ccrs.PlateCarree(),
                          cmap='viridis', alpha=0.6)  # 使用 'viridis' 配色方案，降低透明度

    # 添加颜色条
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('Cost')

    # 绘制动态障碍物的位置和活动范围
    for idx, (lat, lon) in enumerate(dynamic_obstacles):
        ax.plot(lon, lat, 'rx', markersize=8)
        # 绘制动态障碍物的活动范围，颜色根据距离变化
        for radius, alpha in zip([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]):
            ax.add_patch(
                Ellipse(
                    (lon, lat), width=radius * 2, height=radius * 2,
                    edgecolor='red', facecolor='none', linestyle='--', alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            )

    # 绘制静态障碍物的位置和缓冲区
    for idx, (lat, lon) in enumerate(static_obstacles):
        ax.plot(lon, lat, 'bo', markersize=5)
        # 绘制静态障碍物的缓冲区，颜色根据距离变化
        for radius, alpha in zip([0.05, 0.1], [0.3, 0.2]):
            ax.add_patch(
                Ellipse(
                    (lon, lat), width=radius * 2, height=radius * 2,
                    edgecolor='blue', facecolor='none', linestyle='--', alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            )

    # 设置标题和轴标签
    ax.set_title("Cost Field Visualization", fontsize=16)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    # 设置经纬度刻度
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xticks(np.arange(118, 127, 1))
    ax.set_yticks(np.arange(10, 17, 1))
    ax.set_xticklabels([f"{int(i)}°E" for i in np.arange(118, 127, 1)], fontsize=10)
    ax.set_yticklabels([f"{int(i)}°N" for i in np.arange(10, 17, 1)], fontsize=10)

    # 添加自定义图例
    create_custom_legend(ax)

    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 加载陆地多边形
    land_prepared = load_land_polygons('10m')  # 使用更高分辨率的陆地数据

    resolution = 800  # 增加分辨率以提高路径精度
    start_coord = (14.0, 119.0)  # 起点(纬度, 经度)
    goal_coord = (10.0, 122.0)  # 终点(纬度, 经度)

    # 初始化代价场（向量化）
    print("初始化代价场...")
    lats, lons, cost_field = initialize_cost_field_vectorized(
        (10, 16), (118, 126), resolution, dynamic_obstacles, static_obstacles, land_prepared
    )
    print("代价场初始化完成。")

    # 创建陆地掩码
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T
    mask_land = np.array([land_prepared.contains(geom.Point(lon, lat)) for lon, lat in points]).reshape(lon_grid.shape)

    # 打印代价场的统计信息
    ocean_cost_min = np.nanmin(cost_field[~mask_land])
    ocean_cost_max = np.nanmax(cost_field[~mask_land])
    print(f"海洋区域代价场最小值: {ocean_cost_min}")
    print(f"海洋区域代价场最大值: {ocean_cost_max}")

    # 验证起点和终点是否在陆地上
    start_point = geom.Point(start_coord[1], start_coord[0])
    goal_point = geom.Point(goal_coord[1], goal_coord[0])
    print(f"起点在陆地上: {land_prepared.contains(start_point)}")
    print(f"终点在陆地上: {land_prepared.contains(goal_point)}")

    if land_prepared.contains(start_point):
        print("起点位于陆地上，请选择海洋区域作为起点。")
    if land_prepared.contains(goal_point):
        print("终点位于陆地上，请选择海洋区域作为终点。")

    # 确保起点和终点在海洋区域
    if not land_prepared.contains(start_point) and not land_prepared.contains(goal_point):
        # 执行FMM
        print("执行快速扩展方法 (FMM)...")
        times, goal_idx, start_idx = fast_marching_method(
            lats, lons, cost_field, start_coord, goal_coord
        )
        print("快速扩展方法 (FMM) 完成。")

        # 重建路径
        print("重建路径...")
        path = reconstruct_path(times, start_idx, goal_idx)
        if len(path) > 0:
            print(f"路径重建完成，路径长度: {len(path)}")
        else:
            print("未找到路径。")

        # 绘制路径图
        print("绘制路径图...")
        plot_path(
            lats, lons, cost_field, path, dynamic_obstacles, static_obstacles, start_coord, goal_coord
        )
        print("路径图绘制完成。")
    else:
        print("起点或终点位于陆地上，无法执行路径规划。")

    # 绘制代价场可视化图
    print("绘制代价场可视化图...")
    plot_cost_field(lats, lons, cost_field, dynamic_obstacles, static_obstacles)
    print("代价场可视化图绘制完成。")
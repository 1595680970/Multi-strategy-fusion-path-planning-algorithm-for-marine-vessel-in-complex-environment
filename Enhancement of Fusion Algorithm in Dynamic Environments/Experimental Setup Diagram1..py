import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse

# 模拟船舶的经纬度（根据实际需求修改）
pirate_ships = [
    (12.01535, 123.00842),  # 海盗船1
    (11.99912, 121.00892),  # 海盗船2
    (11.33315, 120.5530),   # 海盗船3
    (13.6163, 120.99792)    # 海盗船4
]

# 模拟其它船舶的经纬度（根据实际需求修改）
other_ships = [
    (12.77668, 120.6189),   # 其他船舶1
    (11.5824, 120.50353),
    (12.88988, 122.6239),
    (11.25725, 120.48157),
    (12.0586, 120.4596),
    (11.48488, 121.05835),
    (13.35827, 120.00915),
    (12.77668, 120.6189),
    (13.13765, 121.81092),
    (13.36365, 119.92127),
    (13.41743, 119.98168),
    (13.37440, 119.78943),
    (12.99763, 120.16845),
    (13.60018, 121.78345),
    (11.95045, 120.44312),
    (12.06400, 121.30005),
    (11.02942, 120.49805),
    (11.92340, 120.08057),
    (10.75795, 120.54748),
    (11.02398, 120.50903),
    (10.91543, 120.1355),
    (10.32853, 120.40467),
    (10.57865, 120.531),
    (12.53395, 120.3772),
    (10.96428, 120.50903),
    (10.93715, 120.15198),
    (11.83143, 120.48157),
    (11.35483, 121.8274),
    (12.72277, 120.22338),
    (12.98687, 120.19592),
    (12.88448, 118.57543),
    (12.71737, 118.70728),
    (10.76338, 122.79418),
    (13.36903, 118.91602),
    (12.12888, 124.40918),
    (11.28977, 123.3435),
]

# 创建绘图
fig = plt.figure(figsize=(14, 14))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置绘图范围，确保覆盖研究区域，适当放大比例尺
ax.set_extent([118, 126, 10, 16])  # 经度118°E到126°E，纬度10°N到16°N

# 绘制海岸线和边界
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 绘制其他船舶（绿色圆圈表示）
for lat, lon in other_ships:
    ax.plot(lon, lat, marker='o', color='green', markersize=5)

# 绘制海盗船（红色叉表示）
for lat, lon in pirate_ships:
    ax.plot(lon, lat, marker='x', color='red', markersize=8)
    # 添加海盗船的活动范围：用不同颜色的圈表示
    for radius in [0.1, 0.2, 0.3]:  # 半径范围可以根据实际情况调整
        ax.add_patch(
            Ellipse(
                (lon, lat), width=radius * 2, height=radius * 2,
                edgecolor='red', facecolor='none', linestyle='--', alpha=0.5
            )
        )

# 设置标题
ax.set_title("Experimental Setup Diagram for Experiment 1", fontsize=14)

# 设置经纬度刻度
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

# 设置四周经纬度标注
ax.set_xticks(np.arange(118, 127, 1))  # 经度范围
ax.set_yticks(np.arange(10, 17, 1))    # 纬度范围
ax.set_xticklabels([f"{int(i)}°E" for i in np.arange(118, 127, 1)], fontsize=10)
ax.set_yticklabels([f"{int(i)}°N" for i in np.arange(10, 17, 1)], fontsize=10)

# 设置图例
ax.plot([], [], marker='o', color='green', linestyle='None', markersize=5, label="Static Obstacles")
ax.plot([], [], marker='x', color='red', linestyle='None', markersize=8, label="Dynamic Obstacles")
ax.add_patch(Ellipse((0, 0), width=0.3, height=0.3, edgecolor='red', facecolor='none', linestyle='--', alpha=0.5))
ax.plot([], [], linestyle='--', color='red', label="Dynamic Obstacles Activity Range")
legend = ax.legend(loc='upper right', fontsize=10, frameon=False)
for label in legend.get_texts():
    label.set_fontsize(12)
    label.set_color('black')

# 显示图形
plt.show()

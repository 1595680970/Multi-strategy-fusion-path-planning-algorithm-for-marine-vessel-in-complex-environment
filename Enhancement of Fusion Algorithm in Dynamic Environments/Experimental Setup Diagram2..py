import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 创建绘图
fig = plt.figure(figsize=(14, 14))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置绘图范围（12-13°N, 123-124°E）
ax.set_extent([123, 124, 12, 13])

# 添加底图要素：陆地、海洋、海岸线、边界
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='bisque')
ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

# 标题
ax.set_title("Experimental Setup Diagram for Experiment 2", fontsize=16)

# 设置经纬度网格
# 这里将经纬度间隔细分为每0.2度
lon_ticks = np.arange(123, 124.1, 0.2)
lat_ticks = np.arange(12, 13.1, 0.2)

ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

ax.set_xticklabels([f"{lon:.1f}°E" for lon in lon_ticks], fontsize=10)
ax.set_yticklabels([f"{lat:.1f}°N" for lat in lat_ticks], fontsize=10)

# 添加网格线（使用较浅的颜色和虚线）
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

plt.show()

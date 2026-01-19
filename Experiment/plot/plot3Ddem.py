import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dem_3d(tif_path):
    with rasterio.open(tif_path) as src:
        dem = src.read(1).astype(float)

        # 将无效值设为 nan
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

        # 获取坐标格网
        height, width = dem.shape
        x = np.arange(0, width)
        y = np.arange(0, height)
        X, Y = np.meshgrid(x, y)

    # 绘制 3D 表面
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, dem, cmap='terrain')

    ax.set_title("3D DEM Surface Plot")
    plt.show()


if __name__ == "__main__":
    plot_dem_3d(r"F:\datasets\DEM\z_data\z_30m_12regions\region3_tiny_512\patch_00199.tif")

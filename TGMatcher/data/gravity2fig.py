import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from scipy import ndimage
from scipy.interpolate import griddata

# CSV_PATH = "gravity.csv"       # 输入 CSV
# OUT_FULLMAP = "gravity_map.png"  # 输出完整图像
# PATCH_DIR = "gravity_patches"    # 裁剪 patch 保存路径

# MAP_RES = 2048       # 生成重力场图像分辨率（2048×2048）
# PATCH_SIZE = 256      # patch 大小
# PATCH_STRIDE = 128    # 滑动步长

def load_csv(csv_path):
    try:
        import polars as pl
        print("Using Polars for fast CSV reading...")
        df = pl.read_csv(csv_path)
        lat = df["Latitude"].to_numpy()
        lon = df["Longitude"].to_numpy()
        gval = df["gravity_anomaly"].to_numpy()
    except:
        import pandas as pd
        print("Polars not found, using pandas...")
        df = pd.read_csv(csv_path)
        lat = df["Latitude"].values
        lon = df["Longitude"].values
        gval = df["gravity_anomaly"].values

    print(f"Loaded {len(lat)} gravity points.")
    return lat, lon, gval


def rasterize_and_fill(lat, lon, g, res, out_path):
    """
    使用 histogram2d 做初步栅格化，
    再用 griddata(method='linear') 做线性插值，
    最后用高斯滤波平滑。
    """
    print("Interpolating using linear method and saving gravity map...")

    # grid范围
    lat_min, lat_max = 60, 70
    lon_min, lon_max = -120, -80
    # lat_range = lat_max - lat_min
    # lon_range = lon_max - lon_min
    k = 4 # lon_range / lat_range # x/y
    res_x = res
    res_y = int(res / k) 
    # 生成网格
    grid_lat = np.linspace(lat_min, lat_max, res_y)
    grid_lon = np.linspace(lon_min, lon_max, res_x)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    # 使用 linear 插值(griddata)
    points = np.vstack([lon, lat]).T
    values = g.astype(np.float32)
    print("Using griddata (linear)...")
    grid_g = griddata(
        points, 
        values, 
        (grid_x, grid_y), 
        method="linear"
    )

    # 对 linear 无法覆盖的区域，用 nearest 填补
    nan_mask = np.isnan(grid_g)
    if np.any(nan_mask):
        print("Filling missing regions using nearest interpolation...")
        grid_g[nan_mask] = griddata(
            points, 
            values, 
            (grid_x[nan_mask], grid_y[nan_mask]), 
            method="nearest"
        )

    # 高斯平滑
    sigma = max(1, res / 1024)
    smooth = ndimage.gaussian_filter(grid_g, sigma=sigma)
    # 归一化
    mn, mx = smooth.min(), smooth.max()
    norm = (smooth - mn) / (mx - mn + 1e-12)
    h, w = norm.shape
    print(f"global img shape is {w} x {h}")
    # 绘制主图 + 等重力线
    plt.figure(figsize=(8, 8))
    plt.imshow(
        norm, 
        origin="lower", 
        cmap="gray", 
        extent=[lon_min, lon_max, lat_min, lat_max]
    )
    plt.colorbar(label="Normalized Gravity Anomaly")
    # 添加等值线（虚线）
    # contour_levels = np.linspace(0, 1, 15)  # 20 条等值线
    # cs = plt.contour(
    #     grid_x,
    #     grid_y, 
    #     norm,
    #     levels=contour_levels,
    #     colors="black",
    #     linewidths=0.1,
    #     # linestyles="--"
    # )  # 虚线

    # plt.clabel(cs, inline=True, fontsize=6) # 显示数字
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.close()
    return norm


def make_patches(img, out_dir, size=256, stride=128):
    os.makedirs(out_dir, exist_ok=True)
    print("Cropping patches...")
    h, w = img.shape
    k = 0
    for y in range(0, h - size, stride):
        for x in range(0, w - size, stride):
            patch = img[y:y + size, x:x + size]
            imsave(f"{out_dir}/patch_{k}.png", patch, cmap="gray")
            k += 1

    print(f"Saved {k} patches into {out_dir}")


if __name__ == "__main__":
    root_path = r"E:\.Project\#Cross_Modal\TerrainGravityMatcher\data\gravity"
    lat, lon, gval = load_csv(
        csv_path=os.path.join(root_path, "filtered_181-192_merged_yichang_labeled.csv")
    )
    grid_g = rasterize_and_fill(
        lat=lat,
        lon=lon, 
        g=gval, 
        res=2048,
        out_path=os.path.join(root_path, "global_2048_gray.png")
    )
    make_patches(
        img=grid_g, 
        out_dir=os.path.join(root_path, "2048_patch_256_gray"),
        size=256,
        stride=128
    )
    print("All done.")

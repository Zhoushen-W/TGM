import os
import cv2
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import map_coordinates


def sample_track_by_distance(p0, p1, x):
    """
    沿航迹每x米采一个ping
    """
    v = p1 - p0
    L = np.linalg.norm(v)
    t_hat = v / L # 切向量
    s_vals = np.arange(0, L, x)
    points = p0[None, :] + s_vals[:, None] * t_hat
    return points, s_vals, t_hat

def sample_depth_at_points(dem, transform, points_xy):
    xs, ys = points_xy[:, 0], points_xy[:, 1]
    cols, rows = ~transform * (xs, ys)
    z = map_coordinates(
        dem,
        [rows, cols],
        order=1,
        mode="nearest"
    )
    h = -z  # 返回深度为正
    return h

def create_multibeam_strip(
    data_path,
    start_coord,
    end_coord,
    delta_x,
    N_beams=681,
    resolution=30.0,
    swath_factor=1.2,
    noise_ratio=0.005,
):
    """
    生成一条航迹上的多波束图像(保持给定分辨率)
    Args:
        data_path: DEM路径
        start_coord: 起点(EPSG:3996)
        end_coord: 终点
        delta_x: 采样间隔(m)
        N_beams: 波束数量
        resolution: 给定分辨率
        swath_factor: 单边采样宽度 / 深度
        noise_ratio: 多波束精度
    Returns:
        strip: 条带
        valid: 非NaN值掩码
    """
    with rasterio.open(data_path) as ds:
        dem = ds.read(1)
        transform = ds.transform
    p0 = np.array(start_coord, float)
    p1 = np.array(end_coord, float)
    ping_xy, s_vals, t_hat = sample_track_by_distance(p0, p1, delta_x)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    h_vals = sample_depth_at_points(dem, transform, ping_xy)

    Ns = int(np.floor(s_vals[-1] / resolution))
    h_max = np.nanmax(h_vals)
    swath_max = swath_factor * 2 * h_max
    Nn = int(swath_max // resolution)
    strip = np.full((Ns, Nn), np.nan, dtype=np.float32)

    for i, (p, s, h) in enumerate(zip(ping_xy, s_vals, h_vals)):
        row = int(s // resolution)
        if row < 0 or row >= Ns:
            continue
        half_swath = swath_factor * h
        n_beam = np.linspace(-half_swath, half_swath, N_beams)
        xy = p[None, :] + n_beam[:, None] * n_hat
        z_beam = -sample_depth_at_points(dem, transform, xy)
        sigma = noise_ratio * h
        z_beam += np.random.normal(0, sigma, size=z_beam.shape)
        n_grid = (np.arange(Nn) - Nn // 2) * resolution
        z_grid = np.interp(n_grid, n_beam, z_beam, np.nan, np.nan)
        strip[row, :] = z_grid
    return strip


def show_strip(strip, strip_save_dir=None):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        strip,
        origin="upper", # 航迹方向向下
        cmap="terrain" #"viridis"
    )
    plt.colorbar(im, label="Depth (m)")
    plt.xlabel("Across-track (pixel @30m)")
    plt.ylabel("Along-track (pixel @30m)")
    plt.title("Simulated Multibeam Strip")
    if strip_save_dir is None:
        strip_save_dir = "./Experiment/localization/fig/strip.svg"
    plt.savefig(strip_save_dir, bbox_inches='tight', pad_inches=0)


def sample_multibeam_and_dem_patches(
    dem_path,
    start_coord,
    end_coord,
    delta_x=15,
    mb_patch_size=256,
    mb_patch_stride=64,
    dem_patch_size=1024,
    dem_subpatch_size=256,
    dem_subpatch_stride=128,
    strip_save_dir=None,
    save_dir=None,
    N_beams=681,
    resolution=30.0,
    swath_factor=1.2,
    noise_ratio=0.005
):
    """
    1. 用create_multibeam_strip合成多波束条带(带噪声)
    2. 在条带上以mb_patch_stride步长裁切mb_patch_size的多波束patch,记录中心行号
    3. 每采两张patch,取它们中心行号的中点,映射到航迹物理坐标,再映射到DEM像素坐标,采DEM大patch并裁切小patch
    4. 用DEM大patch分布归一化对应的两张多波束patch
    """
    assert save_dir is not None, "save_dir should not be None"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/beam_images", exist_ok=True)
    os.makedirs(f"{save_dir}/beam_meta", exist_ok=True)

    # 1. 合成多波束条带
    strip = create_multibeam_strip(
        dem_path,
        start_coord,
        end_coord,
        delta_x=delta_x,
        N_beams=N_beams,
        resolution=resolution,
        swath_factor=swath_factor,
        noise_ratio=noise_ratio
    )
    show_strip(strip, strip_save_dir)
    H, W = strip.shape

    # 2. 在条带上采样多波束patch
    mb_patch_rows = []
    mb_patches = []
    for row in range(0, H, mb_patch_stride):
        if H - row < mb_patch_size:
            break
        center = W // 2
        left = center - mb_patch_size // 2
        right = left + mb_patch_size
        patch = np.full((mb_patch_size, mb_patch_size), np.nan, dtype=np.float32)
        # 计算实际可用区域
        row_start = row
        row_end = min(row + mb_patch_size, H)
        col_start = max(left, 0)
        col_end = min(right, W)
        patch_row_start = 0 if row >= 0 else -row
        patch_col_start = 0 if left >= 0 else -left
        patch[patch_row_start:patch_row_start + (row_end - row_start), patch_col_start:patch_col_start + (col_end - col_start)] = \
            strip[row_start:row_end, col_start:col_end]
        mb_patches.append(patch)
        mb_patch_rows.append(row + mb_patch_size // 2)

    # 3. 每张patch,采DEM大patch并裁切小patch
    # 需要映射条带中心行号到物理坐标,再到DEM像素坐标
    # s_vals与strip的行号一一对应
    p0 = np.array(start_coord, float)
    p1 = np.array(end_coord, float)
    _, s_vals, t_hat = sample_track_by_distance(p0, p1, resolution * mb_patch_stride)
    
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
        transform = ds.transform
        dem_H, dem_W = dem.shape

    for i in tqdm(range(len(mb_patches))):
        row = mb_patch_rows[i]
        # 行号转物理距离
        if i >= len(s_vals):
            continue
        s = s_vals[i]
        mid_xy = p0 + s * t_hat
        # 物理坐标转DEM像素
        xs, ys = mid_xy[0], mid_xy[1]
        col, row = ~transform * (xs, ys)
        col, row = int(col), int(row)
        left = col - dem_patch_size // 2
        top = row - dem_patch_size // 2
        dem_patch = dem[max(top, 0):min(top + dem_patch_size, dem_H), max(left, 0):min(left + dem_patch_size, dem_W)]
        patch = np.full((dem_patch_size, dem_patch_size), np.nan, dtype=np.float32)
        h, w = dem_patch.shape
        patch[:h, :w] = dem_patch
        vmin, vmax = np.nanpercentile(patch, 5), np.nanpercentile(patch, 95)

        # 裁切DEM小patch
        for r in range(0, dem_patch_size - dem_subpatch_size + 1, dem_subpatch_stride):
            for c in range(0, dem_patch_size - dem_subpatch_size + 1, dem_subpatch_stride):
                subpatch = patch[r:r + dem_subpatch_size, c:c + dem_subpatch_size]
                sub_vmin, sub_vmax = np.nanpercentile(subpatch, 2), np.nanpercentile(subpatch, 98)
                subpatch_norm = np.clip((subpatch - sub_vmin) / (sub_vmax - sub_vmin + 1e-7), 0, 1)
                subpatch_norm[np.isnan(subpatch)] = 0
                subpatch_img = (subpatch_norm * 255).astype(np.uint8)
                global_row = top + r
                global_col = left + c
                os.makedirs(f"{save_dir}/dem_images/{i:04d}", exist_ok=True)
                os.makedirs(f"{save_dir}/dem_meta/{i:04d}", exist_ok=True)
                img_path = os.path.join(save_dir, "dem_images", f"{i:04d}", f"dem_{i:04d}_{r:03d}_{c:03d}.png")
                cv2.imwrite(img_path, subpatch_img)
                meta = {"coord": (global_row, global_col)}
                torch.save(meta, os.path.join(save_dir, "dem_meta", f"{i:04d}", f"dem_{i:04d}_{r:03d}_{c:03d}.pt"))

        # 归一化对应的两张多波束patch
        
        mb_patch = mb_patches[i]
        mask = np.zeros((mb_patch_size, mb_patch_size), dtype=np.float32)
        valid = ~np.isnan(mb_patch)
        mask[valid] = 1
        mb_sub_vmin, mb_sub_vmax = np.nanpercentile(mb_patch, 2), np.nanpercentile(mb_patch, 98)
        # if mb_sub_vmax - mb_sub_vmin < 300:
        #     mb_sub_vmin, mb_sub_vmax = vmin, vmax
        mb_patch_norm = np.clip((mb_patch - mb_sub_vmin) / (mb_sub_vmax - mb_sub_vmin + 1e-7), 0, 1)
        mb_patch_norm[np.isnan(mb_patch)] = 0
        mb_patch_img = (mb_patch_norm * 255).astype(np.uint8)
        img_path = os.path.join(save_dir, "beam_images", f"beam_{i:04d}.png")
        cv2.imwrite(img_path, mb_patch_img)
        meta = {"prob": torch.from_numpy(mask).float()}
        torch.save(meta, os.path.join(save_dir, "beam_meta", f"beam_{i:04d}.pt"))

    print(f"采样完成, 结果保存在{save_dir}")


def visualize_matching_trajectory(
    dem_path,
    start_coord,
    end_coord,
    match_result,
    beam_num,
    strip_shape,
    patch_size=256,
    stride=64,
    save_path=None,
):
    """
    在DEM大图上可视化:
    - 红线：真实航迹
    - 蓝点: 多波束strip中心线采样点经tfm变换后在DEM全局的像素坐标
    Args:
        dem_path: DEM大图路径
        start_coord, end_coord: strip轨迹起止地理坐标
        transform_matrices: list of [1,2,3] tensor, DEMpatch=M*beam_patch
        idces: DEM patch在数据集中的编号
        strip_shape: (H, W) of strip (多波束图像)
        patch_size, stride: DEM patch参数
        save_path: 保存路径
        prefix: DEM patch命名前缀
        dem_patch_dir: DEM patch图像目录
    """
    import re
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
        transform = ds.transform
    H, W = dem.shape
    dem_crop = dem[H // 2:, W // 2:]
    plt.figure(figsize=(10, 10))
    plt.imshow(dem_crop, cmap='terrain', origin='upper')

    # 真实航迹
    p0 = np.array(start_coord)
    p1 = np.array(end_coord)
    n_points = 50
    track = np.linspace(p0, p1, n_points)
    cols, rows = ~transform * (track[:, 0], track[:, 1])
    plt.plot(cols - W // 2, rows - H // 2, 'r-', label='True Trajectory', linewidth=2)

    strip_H, strip_W = strip_shape
    n_inner = 32   # 每张 patch 内采样点数
    matched_cols, matched_rows = [], []
    print("trajectory is creating ... ")

    for n in range(beam_num):
        ref = match_result[n]
        tfm = ref["tfm"]
        patch_coord = ref["coord"]
        tfm = tfm[0].cpu().numpy() if isinstance(tfm, torch.Tensor) else tfm
        patch_coord = patch_coord
        dem_patch_y0, dem_patch_x0 = patch_coord

        # 每张 patch 内，沿 strip 中心线采样多个点
        if n < beam_num - 1:
            max_row = strip_H // 2
        else:
            max_row = strip_H # 防止重叠
        strip_rows = np.linspace(0, max_row - 1, n_inner)
        for s_row in strip_rows:
            pt_beam = np.array([strip_W // 2, int(s_row), 1.0])
            pt_dem_patch = tfm @ pt_beam
            dem_x = dem_patch_x0 + pt_dem_patch[0]
            dem_y = dem_patch_y0 + pt_dem_patch[1]
            # 右下角视图
            matched_cols.append(dem_x - W // 2)
            matched_rows.append(dem_y - H // 2)
    plt.plot(matched_cols, matched_rows, 'b-', label='Matched Trajectory', linewidth=1)
    plt.legend()
    plt.title('Trajectory Visualization')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

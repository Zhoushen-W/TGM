import rasterio
from rasterio.windows import Window
import cv2
import os
import numpy as np
from tqdm import tqdm


def DEM_Scissor(input_fig, output_dir, patch_size=512, stride=256, i=None):
    """
    将高分辨率DEM剪切

    Args:
        input_fig: 输入图像名称
        output_dir: 输出文件夹路径
        patch_size: 剪切后的图像分辨率
        stride: 步长, 当stride < patch_size时产生重叠
    """
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(input_fig) as src:
        width = src.width
        height = src.height
        print(f"原始尺寸: {width}x{height}")
        patch_id = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # 防止越界
                if x + patch_size > width or y + patch_size > height:
                    continue
                window = Window(x, y, patch_size, patch_size)
                patch = src.read(1, window=window)  # 读取单通道（高程）
                # 忽略全0或无效区域（可选）
                if np.all(patch == 0):
                    continue
                # 保存为tif或npy
                if i is not None:
                    out_path = os.path.join(output_dir, f"patch_{patch_id:05d}_{i}.tif")
                else:
                    out_path = os.path.join(output_dir, f"patch_{patch_id:05d}.tif")
                profile = src.profile
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": rasterio.windows.transform(window, src.transform)
                })
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(patch, 1)
                patch_id += 1

    print(f"已生成 {patch_id} 个裁剪块, 保存在 {output_dir}/")


def DEM_Processor(input_tif, output_png, enhance=False):
    """
    用于转换DEM到可用灰度图

    Args:
        input_tif: 输入的tif图像
        output_png: 输出的png灰度图
    """
    with rasterio.open(input_tif) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan  # 去除无效值
    # 去除异常值 
    valid = dem[~np.isnan(dem)]
    d_min, d_max = np.percentile(valid, (5, 95)) # 去掉极端值（避免过曝） 
    dem = np.clip(dem, d_min, d_max)
    norm = (dem - d_min) / (d_max - d_min + 1e-7) 
    norm = (norm * 255).astype(np.uint8)
    cv2.imwrite(output_png, norm)


if __name__ == "__main__":
    patch_size = 512
    stride = 128
    NO_DEM = True
    for i in range(5):
        input_tif = f"/home/wenzhoushen/datasets/terrain/z_30m_12regions/z_30m_cppt5c_ibcao_crtd_BinDil_region_{i + 1}.tif"
        output_dir = f"/home/wenzhoushen/datasets/terrain/z_30m_12regions/region{i + 1}_tiny_{patch_size}"
        png_dir = f"/home/wenzhoushen/datasets/terrain/z_30m_12regions/region{i + 1}_png_{patch_size}"
        if NO_DEM:
            DEM_Scissor(
                input_fig=input_tif,
                output_dir=output_dir,
                patch_size=patch_size,
                stride=stride,
                i=i+1,
            )
        # os.makedirs(png_dir, exist_ok=True)
        # tif_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".tif")]
        # for tif_file in tqdm(tif_files):
        #     tif_path = os.path.join(output_dir, tif_file)
        #     png_path = os.path.join(png_dir, tif_file.replace(".tif", f"_{i+1}.png"))
        #     DEM_Processor(
        #         tif_path, 
        #         png_path, 
        #         # enhance=True
        #     )

        # print(f"已将 {len(tif_files)} 个裁剪块转换为灰度图, 输出至: {png_dir}")
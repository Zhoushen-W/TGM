import os 
import torch
from torch.utils.data import Dataset
import cv2
import rasterio
from tqdm import tqdm
import numpy as np

from TGMatcher.utils.warp import get_warp


class TGM_Dataset(Dataset):
    def __init__(self, data_path, img_channel=1):
        self.img_dir = os.path.join(data_path, "images")
        self.meta_dir = os.path.join(data_path, "meta")
        self.img_channel = img_channel
        self.names = sorted(
            [f.replace(".pt", "") for f in os.listdir(self.meta_dir) if f.endswith(".pt")]
        )
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        imA_path = os.path.join(self.img_dir, f"{name}_A.png")
        imB_path = os.path.join(self.img_dir, f"{name}_B.png")
        if self.img_channel == 1:
            im_A = cv2.imread(imA_path, 0)
            im_B = cv2.imread(imB_path, 0)
        else:
            im_A = cv2.imread(imA_path)
            im_B = cv2.imread(imB_path)
        im_A_tensor = torch.from_numpy(im_A / 255.0).float()
        im_B_tensor = torch.from_numpy(im_B / 255.0).float()
        # shape: [c, h, w]
        if self.img_channel == 1:
            im_A_tensor = im_A_tensor.unsqueeze(0).repeat(3, 1, 1)
            im_B_tensor = im_B_tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            im_A_tensor = im_A_tensor.permute(2, 0, 1)
            im_B_tensor = im_B_tensor.permute(2, 0, 1)

        meta = torch.load(
            os.path.join(self.meta_dir, f"{name}.pt"),
            map_location="cpu",
            weights_only=True    
        )

        return {
            "im_A": im_A_tensor,
            "im_B": im_B_tensor,
            "prob_A": meta["prob_A"],
            "prob_B": meta["prob_B"],
            "M_AtoB": meta["M_AtoB"],
            "mode": meta["mode"]
        }


class BuildDataset():
    def __init__(
        self, 
        origin_img_path, 
        dataset_path, 
        transform_num, 
        img_channel=1, 
        mode=None, 
        size=256,
        noise_factor=0.005,
        swath_factor=1.4,
        resolution=30
    ):
        self.origin_img_path = origin_img_path
        self.dataset_path = dataset_path
        self.min_transform_num = transform_num
        self.img_channel = img_channel
        self.mode = mode
        self.size = size
        self.noise_factor = noise_factor
        self.swath_factor = swath_factor
        self.resolution = resolution

    def aug_rotate(self, angle, size):
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return M 

    def aug_translate(self, tx, ty):
        M = np.float32([
            [1, 0, tx], 
            [0, 1, ty]
        ])
        return M
    
    def aug_scale(self, scale, size):
        M = np.float32([
            [scale, 0, (1 - scale) * size / 2], 
            [0, scale, (1 - scale) * size / 2]
        ])
        return M
    
    def aug_affine(self, min_transform_num, ops, size):
        num = max(min_transform_num, np.random.randint(min_transform_num, len(ops) + 1))
        ops_selected = np.random.choice(ops, size=num, replace=False)
        tot_M_AtoB = np.float32([[1, 0, 0], [0, 1, 0]])
        for op in ops_selected:
            if op == "rotate":
                M_AtoB = self.aug_rotate(np.random.uniform(-90, 90), size)
            elif op == "translate":
                M_AtoB = self.aug_translate(
                    np.random.uniform(-75, 75), 
                    np.random.uniform(-75, 75)
                )
            elif op == "scale":
                M_AtoB = self.aug_scale(np.random.uniform(0.98, 1.02), size)
            # 这里的写法是左乘链, B = M_k * M_k-1 * ... * M_1 * A
            tot_M_AtoB = M_AtoB @ np.vstack([tot_M_AtoB, [0, 0, 1]])
            # tot_M_AtoB = tot_M_AtoB[:2, :]
        return np.vstack([tot_M_AtoB, [0, 0, 1]])

    def crop_center(self, img):
        # 在大图的中间截取im_A和im_B
        s = (img.shape[0] - self.size) // 2
        return img[s:s + self.size, s:s + self.size]

    def fullM_to_patchM(self, full_M):
        # 仿射变换, 把矩阵M转换到imAB的坐标下
        offset = self.size / 2
        T = np.float32(
            [[1, 0, offset],
             [0, 1, offset],
             [0, 0, 1]]
        )
        T_inv = np.float32(
            [[1, 0, -offset],
             [0, 1, -offset],
             [0, 0, 1]]
        )
        return T_inv @ full_M @ T

    def normalize_dem_percentile(self, dem, pmin=2, pmax=98):
        vmin = np.percentile(dem, pmin)
        vmax = np.percentile(dem, pmax)
        dem_n = (dem - vmin) / (vmax - vmin)
        dem_n = np.clip(dem_n, 0.0, 1.0)
        return (dem_n * 255.0).astype(np.uint8), vmin, vmax

    def build_swath_mask(self, dem, swath_factor=1.4):
        """
        dem: (H, W) float DEM(未归一化)
        """
        H, W = dem.shape
        cx = W // 2
        mask = np.zeros_like(dem, dtype=np.uint8)
        center_line = dem[:, cx]  # 每一行中心高程
        for y in range(H):
            h = center_line[y]
            half_w = int(abs(swath_factor * h) / self.resolution)
            x0 = max(0, cx - half_w)
            x1 = min(W, cx + half_w + 1)
            mask[y, x0:x1] = 1
        return mask.astype(np.float32)
    
    def add_rowwise_noise(self, dem, noise_factor=0.005):
        """
        dem: float DEM(未归一化)
        """
        H, W = dem.shape
        cx = W // 2
        noisy = dem.copy()
        center_line = dem[:, cx]
        for y in range(H):
            h = center_line[y]
            sigma = noise_factor * abs(h)
            if sigma > 0:
                noise = np.random.normal(0.0, sigma, size=W)
                noisy[y] += noise
        return noisy


    def make_pair(self):
        img_path = self.origin_img_path
        min_transform_num = self.min_transform_num
        save_path = self.dataset_path
        os.makedirs(save_path, exist_ok=True)
        im_dir = os.path.join(save_path, "images")
        meta_dir = os.path.join(save_path, "meta")
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        ops = ["rotate", "translate", "scale"]
        imgs_A = [i for i in os.listdir(img_path) if i.endswith(".tif")]

        for i in tqdm(imgs_A):
            full_img_path_A = os.path.join(img_path, i)
            full_img_A = rasterio.open(full_img_path_A).read(1).astype(np.float32)
            h, w = full_img_A.shape[:2]
            assert h == w, "Image shape should satisfy: h = w"

            im_A = self.crop_center(full_img_A)
            im_A_norm, _, _ = self.normalize_dem_percentile(im_A)
            M_AtoB_full = self.aug_affine(min_transform_num, ops, h)
            M_AtoB = self.fullM_to_patchM(M_AtoB_full)[:2, :]
            warped_full_img = cv2.warpAffine(
                full_img_A,
                M_AtoB_full[:2, :],
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            im_B = self.crop_center(warped_full_img).astype(np.float32)
            _, vmin_B, vmax_B = self.normalize_dem_percentile(im_B)
            h_range = vmax_B - vmin_B
            if h_range >= 300:
                im_B = self.add_rowwise_noise(im_B, noise_factor=self.noise_factor)
            im_B_norm, _, _ = self.normalize_dem_percentile(im_B)
            swath_mask = self.build_swath_mask(im_B, self.swath_factor)
            im_B_norm = (im_B_norm * swath_mask).astype(np.uint8)

            warped_prob = cv2.warpAffine(
                np.ones_like(im_A, dtype=np.uint8),
                M_AtoB,
                (self.size, self.size),
                flags=cv2.INTER_NEAREST,
                borderValue=0
            )
            prob_B = self.crop_center(warped_prob).astype(np.float32)
            prob_B = prob_B * swath_mask
            prob_B_tensor = torch.from_numpy(prob_B).float()
            M_AtoB_tensor = torch.from_numpy(M_AtoB).float()
            # M_full_tensor = torch.from_numpy(M_AtoB_full[:2, :]).float()

            x2 = get_warp(
                M_AtoB=M_AtoB_tensor.unsqueeze(0),
                H=self.size,
                W=self.size
            ).squeeze(0)
            dx, dy = x2[..., 0], x2[..., 1]
            prob_A_tensor = (
                (dx >= -1 + 1 / self.size) & (dx <= 1 - 1 / self.size) &
                (dy >= -1 + 1 / self.size) & (dy <= 1 - 1 / self.size)
            ).float()
            ix = ((dx + 1) * 0.5 * (self.size - 1)).long()
            iy = ((dy + 1) * 0.5 * (self.size - 1)).long()
            valid_swath_A = torch.zeros_like(prob_A_tensor)
            inside = (
                (ix >= 0) & (ix < self.size) &
                (iy >= 0) & (iy < self.size)
            )
            valid_swath_A[inside] = torch.from_numpy(swath_mask)[iy[inside], ix[inside]]
            prob_A_tensor = prob_A_tensor * valid_swath_A

            cv2.imwrite(os.path.join(im_dir, i.replace(".tif", "_A.png")), im_A_norm)
            cv2.imwrite(os.path.join(im_dir, i.replace(".tif", "_B.png")), im_B_norm)

            meta = {
                # "im_A": im_A_tensor,
                # "im_B": im_B_tensor,
                # "M_full": M_full_tensor,
                "M_AtoB": M_AtoB_tensor,
                "prob_A": prob_A_tensor,
                "prob_B": prob_B_tensor,
                "mode": self.mode
            }
            torch.save(meta, os.path.join(meta_dir, i.replace(".tif", ".pt")))
        print("Dataset was generated successfully")

if __name__ == "__main__":
    """
    运行前注意调整i和保存地址
    """
    mode = "train"
    assert mode == "train" or mode == "eval", f"mode should be train or eval, but got {mode}"
    rg = range(4) if mode == "train" else range(4, 5)
    for i in rg:
        d = BuildDataset(
            origin_img_path=rf"/home/wenzhoushen/datasets/terrain/z_30m_12regions/region{i+1}_tiny_512",
            dataset_path=rf"/home/wenzhoushen/datasets/tgm_{mode}_dataset",
            transform_num=2,
            img_channel=1,
            mode="terrain"
        )
        d.make_pair()

    data_path = f"/home/wenzhoushen/datasets/tgm_{mode}_dataset"
    sample = 1 if mode == "train" else 5
    data = torch.load(
        f=os.path.join(data_path, "meta", f"patch_00000_{sample}.pt"),
        map_location="cpu",
        weights_only=True
    )
    print(data["prob_A"].shape)
    print(data["prob_B"].shape)
    print(data["M_AtoB"].shape)
    print(data["mode"])

    # im_A: [3, 256, 256]
    # im_B: [3, 256, 256]
    # prob: [256, 256]
    # M_AtoB: [2, 3]

    # file_folder = os.path.join(data_path, "meta")
    # for filename in os.listdir(file_folder):
    #     if filename.endswith("_5_A.png") or filename.endswith("_5_B.png") or filename.endswith("_5.pt"):
    #         file_path = os.path.join(file_folder, filename)
    #         try:
    #             os.remove(file_path)
    #             # print(f"已删除: {file_path}")
    #         except Exception as e:
    #             print(f"删除失败 {file_path}: {e}")
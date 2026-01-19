import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np


@torch.no_grad()
def scale_affine(warp, prob, scale):
    # warp[b, h, w, 2] prob[b, h, w]
    b, h, w = prob.shape
    hs, ws = h // scale, w // scale
    device = prob.device
    x2_grid = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, steps=hs, device=device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, steps=ws, device=device),
        ),
        indexing="ij",
    )
    x2_grid = torch.stack((x2_grid[1], x2_grid[0]), dim=-1)[None].expand(b, hs, ws, 2)
    warp_scaled = F.grid_sample(
        warp.permute(0, 3, 1, 2), # (b, 2, h, w)
        x2_grid,
        mode="bilinear",
        align_corners=False
    ).permute(0, 2, 3, 1) # (b, hs, ws, 2)
    prob_scaled = F.grid_sample(
        prob.unsqueeze(1),
        x2_grid,
        mode="bilinear",
        align_corners=False
    ).squeeze(1) # (b, hs, ws)
    return warp_scaled, prob_scaled


# 检查一下变换是否正确
@torch.no_grad()
def get_warp(M_AtoB, H, W):
    if M_AtoB.shape[-2:] != (2, 3):
        raise ValueError(
            f"Expected transformation matrix shape [B, 2, 3], got {M_AtoB.shape}"
        )
    b, h, w, device = M_AtoB.shape[0], H, W, M_AtoB.device 
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    ) # pixel coords
    ones = torch.ones_like(xs)
    grid_pixel = torch.stack([xs, ys, ones], dim=-1) # (h, w, 3)
    grid_pixel = grid_pixel.view(1, h * w, 3).expand(b, -1, -1).float()
    pad = torch.tensor([0, 0, 1], device=device).view(1, 1, 3)
    M33 = torch.cat([M_AtoB, pad.expand(b, 1, 3)], dim=1) # (b, 3, 3)
    x2_pixel = torch.bmm(grid_pixel, M33.transpose(1, 2))[..., :2].view(b, h, w, 2)
    x_norm = (x2_pixel[..., 0] + 0.5) * 2 / w - 1 # pixel -> nrom
    y_norm = (x2_pixel[..., 1] + 0.5) * 2 / h - 1 # [-1 + 1 / h, 1 - 1 / h]
    x2_norm = torch.stack([x_norm, y_norm], dim=-1)

    return x2_norm # [b, h, w, 2]



# 因为grid_sample采样规则是在A图中采样B的坐标（和warp刚好相反）
# 简单起见改成在图B中，用warp采样A
# def test_warp(im_B, x2, prob=None):
#     assert im_B.dim() == 4
#     assert x2.dim() == 4 and x2.shape[-1] == 2
#     warped_B = F.grid_sample(
#         im_B,
#         x2,
#         mode="bilinear",
#         padding_mode="zeros",
#         align_corners=False
#     )
#     def to_numpy(img):
#         img = img.squeeze(0) # [c, h, w]
#         if img.shape[0] == 1:  # grayscale
#             return img[0].cpu().numpy()
#         else:  # RGB
#             return img.permute(1, 2, 0).cpu().numpy()
#     im_B_np = to_numpy(im_B)
#     warped_np = to_numpy(warped_B)

#     def norm_vis(x):
#         x = x.astype(np.float32)
#         if x.max() > 1.0:
#             x = x / 255.0
#         return np.clip(x, 0, 1)
#     im_B_np = norm_vis(im_B_np)
#     warped_np = norm_vis(warped_np)

#     if prob is not None:
#         prob_np = prob.squeeze(0).cpu().numpy()
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#         axs[2].imshow(prob_np, cmap="jet")
#         axs[2].set_title("Prob / Mask")
#         axs[2].axis("off")
#     else:
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     axs[0].imshow(im_B_np, cmap="gray" if im_B_np.ndim == 2 else None)
#     axs[0].set_title("Original im_B")
#     axs[0].axis("off")
#     axs[1].imshow(warped_np, cmap="gray" if warped_np.ndim == 2 else None)
#     axs[1].set_title("Warped im_A (via x2_norm)")
#     axs[1].axis("off")
#     plt.suptitle("test_warp")
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     M = torch.tensor(
#         [[[2.0, 0.0, -2],
#           [0.0, 2.0, -2]]], 
#         device=device
#     )
#     P = torch.ones(1, 10, 10, device=device)
#     x, prob = get_warp(M, 5, 5, P, "nearest-exact")
#     print(x.float()[0, 1, 1])
#     print(prob.float(), prob.float().shape)


# 最后返回的坐标系不正确！应该用归一化坐标系
# def get_warp(M_AtoB, H, W, prob=None, interpolation_mode="bilinear"):
#     if M_AtoB.shape[-2:] != (2, 3):
#         raise ValueError(
#             f"Expected transformation matrix shape [B, 2, 3], got {M_AtoB.shape}"
#         )
    
#     b, h, w = M_AtoB.shape[0], H, W
#     grid = torch.meshgrid(
#         (
#             torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=M_AtoB.device),
#             torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=M_AtoB.device),
#         ),
#         indexing="ij",
#     )
#     # print(grid)
#     grid = torch.stack((grid[1], grid[0]), dim=-1)[None].expand(b, h, w, 2) # .reshape(b, -1, 2)
#     x = (grid[..., 0] + 1) * 0.5 * (w - 1) # norm -> pixel
#     y = (grid[..., 1] + 1) * 0.5 * (h - 1)
#     grid_pixel = torch.stack([x, y], dim=-1).reshape(b, -1, 2) # pixel coords
#     ones = torch.ones(b, grid_pixel.shape[1], 1, device=M_AtoB.device)
#     grid_homo = torch.cat([grid_pixel, ones], dim=-1)
#     pad = torch.tensor([0, 0, 1], device=M_AtoB.device).view(1, 1, 3)
#     M33 = torch.cat([M_AtoB, pad.expand(b, 1, 3)], dim=1)  # (B,3,3)
#     x2 = torch.bmm(grid_homo, M33.transpose(1, 2))[..., :2].reshape(b, h, w, 2)
#     # x2 = torch.bmm(grid_homo, M_AtoB.transpose(1, 2)).reshape(b, h, w, 2)
#     x_norm = x2[..., 0] / (w - 1) * 2 - 1 # pixel -> norm
#     y_norm = x2[..., 1] / (h - 1) * 2 - 1
#     x2_norm = torch.stack([x_norm, y_norm], dim=-1)

#     if prob is not None:
#         # bs, hs, ws = prob.shape
#         assert h == w, "For every scales, should satisfy H = W"
#         prob = F.interpolate(
#             prob.unsqueeze(1),
#             size=(h, w),
#             mode=interpolation_mode,
#             align_corners=False if interpolation_mode == "bilinear" else None,
#         ).squeeze(1)
#     else:
#         prob = torch.ones(b, h, w, device=M_AtoB.device)

#     return x2_norm, prob
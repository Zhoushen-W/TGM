import torch
import torch.nn.functional as F
import numpy as np

from TGMatcher.data.generate_dataset import TGM_Dataset


@torch.no_grad()
def compute_HM(img, prob=None, patch_size=None, num_bins=16, eps=1e-9, norm_mode=None, a=10, b=1):
    """
    计算基于高程熵的 prob

    Args:
        img : [b, h, w, 3]高程图
        prob: [b, h, w]原始mask(A中哪些像素参与了变换)
        patch_size: 分窗大小(默认256 // 16)
        num_bins: 直方图bin数(把高程值分为多少个区间)
        norm_mode: 选择是否使用sigmoid激活(a > 1, b > 1是增强系数)

    Returns:
        prob_new: [b, h, w]新的prob(float)
    """
    b, c, h, w = img.shape
    device = img.device
    assert h == w, "we assume that image has the same value of h and w"
    if patch_size is None:
        patch_size = h // 16
    elev = img[:, 0, :, :].unsqueeze(1) # [b, 1, h, w]高程值
    patches = F.unfold(
        elev,
        kernel_size=patch_size,
        stride=patch_size,
    ).transpose(1, 2) # [b, patch_num, patch_area]
    b, patch_num, patch_area = patches.shape
    entropy = torch.zeros(b, patch_num, device=device)

    p_min = patches.min(dim=2, keepdim=True).values
    p_max = patches.max(dim=2, keepdim=True).values
    patches_norm = (patches - p_min) / (p_max - p_min + eps)
    bins = torch.clamp((patches_norm * (num_bins - 1)).long(), 0, num_bins - 1)
    hist = torch.zeros(
        size=(b, patch_num, num_bins),
        device=device,
        dtype=torch.float32
    )
    # self[i][j][index[i][j][k]] += src[i][j][k] # if dim == 2
    hist.scatter_add_(
        dim=2,
        index=bins,
        src=torch.ones_like(bins, dtype=torch.float32)
    )
    prob_hist = hist / (hist.sum(dim=2, keepdim=True) + eps)
    entropy = -torch.sum(prob_hist * torch.log(prob_hist + eps), dim=2) # [b, patch_num, 1]

    hp = h // patch_size
    wp = w // patch_size
    entropy = F.interpolate(
        entropy.view(b, 1, hp, wp),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1) # [b, h, w]
    e_min = entropy.amin(dim=(1, 2), keepdim=True)
    e_max = entropy.amax(dim=(1, 2), keepdim=True)
    entropy_norm = (entropy - e_min) / (e_max - e_min + eps)

    if norm_mode is None:
        entropy_norm_prob = entropy_norm ** b
    elif norm_mode == "sigmoid":
        # [0, 1] -> [-0.5, 0.5] -> [-5, 5] 
        entropy_norm_prob = torch.sigmoid(a * (entropy_norm - 0.5)) # sigmoid(4.7) = 0.991
    else:
        raise ValueError("norm_mode except sigmoid is not support currently")

    if prob is not None:
        return entropy_norm_prob * prob
    else:
        return entropy_norm_prob
    

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_path = "./Data/terrain/test_pair"
#     dataset = TGM_Dataset(data_path)
#     sample = 13
#     batch_device = {
#         "im_A": dataset[sample]["im_A"].unsqueeze(0).to(device),
#         "prob": dataset[sample]["prob_A"].unsqueeze(0).to(device),
#     }
#     prob_new = compute_HM(
#         img=batch_device["im_A"],
#         prob=batch_device["prob"],
#         norm_mode="sigmoid",
#         num_bins=32,
#         a=16,
#     )
#     x = 0.99
#     ratio = (prob_new >= x).float().mean(dim=(1, 2)) # [b]
#     for i, r in enumerate(ratio):
#         print(f"Image {i}: {r.item() * 100:.2f}% of pixels >= {x}, with {r.item() * 256**2:.0f} pixels")

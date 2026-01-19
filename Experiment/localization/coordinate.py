import os
import torch
import warnings

from TGMatcher.train.evaluate_tgm import eval_setting
from TGMatcher.data.generate_dataset import TGM_Dataset


def create_grid(h, w, device, b=1):
    grid = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
        ),
        indexing="ij",
    )
    grid = torch.stack((grid[1], grid[0]), dim=-1)[None].expand(b, h, w, 2)
    return grid # [b, h, w, 2]


def select_top_k(flow, certainty, device, top_k_num):
    b, c ,h, w = flow.shape
    flow_reshaped = flow.permute(0, 2, 3, 1).reshape(b, h * w, c)
    certainty_reshaped = certainty[:, 0].reshape(b, h * w)
    grid_reshaped = create_grid(h, w, device=device, b=b).reshape(b, h * w, c)
    selected_indices = []
    for i in range(b):
        cert = certainty_reshaped[i]
        mask = cert > 0.99
        masked_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if masked_idx.numel() >= top_k_num:
            high_cert = cert[masked_idx]
            _, local_top = torch.topk(high_cert, top_k_num)
            top_idx = masked_idx[local_top]
        else:
            _, top_idx = torch.topk(cert, top_k_num)
            warnings.warn(
                f"[select_top_k] Batch {i}: "
                f"only {masked_idx.numel()} points satisfy certainty > 0.99,"
                f"less than {top_k_num} , using global top-k instead.",
                RuntimeWarning
            )
        selected_indices.append(top_idx)
    top_indices = torch.stack(selected_indices, dim=0) # [b, k]
    top_values = torch.gather(certainty_reshaped, 1, top_indices) # [b, k]
    top_indices = top_indices.unsqueeze(-1).expand(-1, -1, c) # [b, k, 2]
    selected_A = torch.gather(grid_reshaped, 1, top_indices) # [b, k, 2]
    selected_B = torch.gather(flow_reshaped, 1, top_indices) # [b, k, 2]
    return selected_A, selected_B, top_values


def estimate_transform_test(data_path, model, device, top_k_num=100):
    dataset = TGM_Dataset(data_path)
    sample = 100
    batch = dataset[sample]
    batch_device = {
        "im_A": batch["im_A"].unsqueeze(0).to(device),
        "im_B": batch["im_B"].unsqueeze(0).to(device),
    }
    corresps = model(batch_device)
    flow = corresps[1]["flow"] # [b, c, h, w]
    certainty = corresps[1]["certainty"]
    M_AtoB = batch["M_AtoB"].unsqueeze(0).to(device) # [b, 2, 3], 真实坐标
    points_A, points_B, top_certainties = select_top_k(flow, certainty, device, top_k_num)
    points_A = (points_A + 1) * 256 / 2 - 0.5
    points_B = (points_B + 1) * 256 / 2 - 0.5
    M_BtoA = weighted_least_square(points_A, points_B, top_certainties)
    diff = eval_transform(M_AtoB, M_BtoA)
    return M_BtoA, diff


def estimate_transform(flow, certainty, device, top_k_num=500):
    points_A, points_B, top_certainties = select_top_k(flow, certainty, device, top_k_num)
    points_A = (points_A + 1) * 256 / 2 - 0.5
    points_B = (points_B + 1) * 256 / 2 - 0.5
    M_BtoA = weighted_least_square(points_A, points_B, top_certainties)
    return M_BtoA


def weighted_least_square(points_A, points_B, weights=None):
    """
    Args:
        points_A: A中选中的点[b, k, 2]
        points_B: B中选中的点[b, k, 2]
        weights: 每个点的权重[b, k] > 0
    Returns:
        M_BtoA: 仿射变换矩阵[b, 2, 3], A = M * B
    """
    b, k, _ = points_A.shape
    if weights is None:
        weights = torch.ones((b, k), device=points_A.device, dtype=points_A.dtype)
    ones = torch.ones((b, k, 1), device=points_B.device, dtype=points_B.dtype)
    B_homo = torch.cat([points_B, ones], dim=-1)  # [b, k, 3]
    W = weights.unsqueeze(-1)  # [b, k, 1]
    B_w = B_homo * W  # [b, k, 3]

    # M = (B^T W B)^{-1} B^T W A
    BTB = torch.matmul(B_w.transpose(1, 2), B_homo)  # [b, 3, 3]
    BTA = torch.matmul(B_w.transpose(1, 2), points_A)  # [b, 3, 2]
    M_BtoA = torch.linalg.solve(BTB, BTA)  # [b, 3, 2]
    M_BtoA = M_BtoA.transpose(1, 2).float() # [b, 2, 3]
    return M_BtoA


def eval_transform(M_AtoB, M_BtoA):
    """
    Args:
        M_AtoB: A到B的变换[b, 2, 3], B = M * A
        M_BtoA: B到A的变换[b, 2, 3], A = M * B
    Returns:
        diff: 二者乘积和单位阵的差异
    """
    if M_AtoB.dtype != M_BtoA.dtype: 
        ValueError(f"we assume this 2 matrices have the same dtype, but got {M_AtoB.dtype} and {M_BtoA.dtype}")
    b = M_AtoB.shape[0]
    ones = torch.zeros((b, 1, 3), device=M_AtoB.device, dtype=M_AtoB.dtype)
    ones[..., 2] = 1
    M_AtoB_homo = torch.cat([M_AtoB, ones], dim=1)
    M_BtoA_homo = torch.cat([M_BtoA, ones], dim=1)
    identity = torch.eye(3, device=M_AtoB.device).unsqueeze(0).expand(b, -1, -1)
    diff = (torch.bmm(M_BtoA_homo, M_AtoB_homo) - identity).norm(dim=(1, 2))
    print(M_AtoB_homo, M_BtoA_homo)
    print(torch.bmm(M_BtoA_homo, M_AtoB_homo))
    return diff


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_BtoA, diff = estimate_transform_test(
        data_path=r"/home/wenzhoushen/datasets/tgm_eval_dataset",
        model=eval_setting(device),
        device=device,
        top_k_num=500
    )
    print(f"tansform diff is {diff.mean().item():.2f}")
    
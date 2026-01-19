import torch
import torch.nn.functional as F

from TGMatcher.utils.warp import get_warp, scale_affine


if __name__ == "__main__":
    data = torch.load(r"F:\datasets\DEM\z_data\train_dataset\patch_00162_1.pt")
    im_A = data["im_A"]
    im_B = data["im_B"]
    prob = data["prob"]
    M = data["M_AtoB"]
    im_A = im_A.cpu().detach().permute(1, 2, 0) # c, h, w -> h, w, c
    im_B = im_B.cpu().detach().permute(1, 2, 0)
    prob_np = prob.cpu().detach().numpy() # h, w

    M_scaled = scale_affine(M.unsqueeze(0), 2)
    warp_scaled, prob_scaled = get_warp(
        M_AtoB=M_scaled,
        H=128,
        W=128,
        prob=prob.unsqueeze(0),
        interpolation_mode="nearest-exact"
    )
    b, h, w, _ = warp_scaled.shape
    gt_warp, gt_prob = get_warp(
        M_AtoB=M.unsqueeze(0),
        H=256,
        W=256,
        prob=prob.unsqueeze(0),
        interpolation_mode="nearest-exact"
    )
    x2s = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w),
        ),
        indexing="ij",
    )
    x2s = torch.stack((x2s[1], x2s[0]), dim=-1)[None].expand(b, h, w, 2)
    x2_s = F.grid_sample(
        gt_warp.permute(0, 3, 1, 2), # (B,2,H,W)
        x2s,
        align_corners=False
    ).permute(0, 2, 3, 1) # (B,h_s,w_s,2)
    with torch.no_grad():
        print(x2_s - warp_scaled)
        

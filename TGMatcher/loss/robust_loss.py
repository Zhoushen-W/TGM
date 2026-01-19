# =============================================================================
# This file is adapted from the RoMa project:
#   https://github.com/Parskatt/RoMa
#
# Original paper:
#   Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg
#   "RoMa: Robust Dense Feature Matching"
#   IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024
#
# Licensed under the MIT License.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math
from einops.einops import rearrange

from TGMatcher.utils.warp import get_warp, scale_affine
from TGMatcher.utils.prob import compute_HM


class Robust_Loss(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=4, # fine_feature里的最大值?
        smooth_mask=False,
        depth_interpolation_mode="bilinear",
        mask_depth_loss=False,
        relative_depth_error_threshold=0.05,
        alpha=1.,
        c=1e-3,
        step=0,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.step = step

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            b, c, h, w = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(c))
            G = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device),
                    torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device),
                ),
                indexing="ij",
            )
            G = torch.stack((G[1], G[0]), dim=-1).reshape(c, 2)
            GT = (G[None, :, None, None, :] - x2[:, None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction="none")[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:, 0], prob)
        # certainty_loss = torch.tensor(0.0, device=x2.device)
        if not torch.any(cls_loss):
            cls_loss = certainty_loss * 0.0 # when prob is 0 everywhere

        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        # wandb.log(losses, step=self.step)
        return losses
    
    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            b, c, h, w = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(c))
            G = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device),
                    torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device),
                )
            )
            G = torch.stack((G[1], G[0]), dim=-1).reshape(c, 2) * offset_scale
            GT = (G[None, :, None, None, :] + flow_pre_delta[:, None] - x2[:, None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduce="none")[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        if not torch.any(cls_loss):
            cls_loss = certainty_loss * 0.0

        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        # wandb.log(losses, step=self.step)
        return losses
    
    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode="delta"):
        epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2 / 256)).float().mean() # how to set the value?
            # wandb.log({"train_pck_05": pck_05}, step=self.step)
        
        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob) # 暂时隔离
        # ce_loss = torch.tensor(0.0, device=x2.device)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x / cs)**2 + 1**2)**(a / 2)
        if not torch.any(reg_loss):
            reg_loss = ce_loss * 0.0
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        # wandb.log(losses, step=self.step)
        return losses

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        scale_weights = {1:1, 2:1, 4:1, 8:1}
        gt_warp = get_warp( # full_res
            M_AtoB=batch["M_AtoB"],
            H=256,
            W=256,
        )
        gt_prob = compute_HM(
            img=batch["im_A"],
            prob=batch["prob_A"],
            num_bins=32,
            norm_mode="sigmoid",
            a=16,
        )
        for scale in scales:
            scale_corresps = corresps[scale]
            (
                scale_certainty, 
                flow_pre_delta, 
                delta_cls, 
                offset_scale, 
                scale_gm_cls, 
                scale_gm_certainty, 
                flow, 
                scale_gm_flow
            ) = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow")
            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                b, _, h, w = scale_certainty.shape
            
            # 需要计算的gt_warp是指图像A中的像素经过变换, 在图像B中对应的坐标
            if scale > 1:
                gt_warp_scaled, gt_prob_scaled = scale_affine(gt_warp, gt_prob, scale) 
                x2 = gt_warp_scaled.float()
                prob = gt_prob_scaled.float()
            else:
                x2 = gt_warp.float()
                prob = gt_prob.float()

            if self.local_largest_scale >= scale:
                p = F.interpolate(
                        prev_epe[:, None],
                        size=(h, w),
                        mode="nearest-exact",
                    )[:, 0]
                prob = prob * (p < (2 / 256) * (self.local_dist[scale] * scale)) # 确保在归一化范围内

            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode="gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss

            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1).detach()
        return tot_loss



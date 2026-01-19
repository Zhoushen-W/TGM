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

import sys
import os
import warnings
from functools import partial
import loguru
import torch
import torch.nn as nn

from TGMatcher.model.encoder import Coarse_Fine_Encoder
from TGMatcher.model.matcher import (
    CosKernel,
    GP,
    ConvRefiner,
    Decoder,
    RegressionMatcher,
)
from TGMatcher.model.Transformer import Transformer_Decoder
from TGMatcher.model.Transformer.layers import Block, MemEffAttention


def tgm_model(
    resolution,
    upsample_preds,
    device=None,
    tgm_weight=None,
    encoder_pretrained=True,
    use_custom_corr=True,
    upsample_res=None,
    sample_thresh=0.05,
    sample_mode="threshold_balanced",
    attenuate_cert=True,
    **kwargs,
):
    # if sys.platform != "linux":
    use_custom_corr = False # only suppored on Linux platform
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)
    if str(device) == "cpu":
        amp_dtype = torch.float32

    # how to set the value of the following parameters?
    gp_dim = 320 # 256?
    feat_dim = 320
    decoder_dim = gp_dim + feat_dim
    cls_to_coords_res = 64
    tf_blocks = nn.Sequential(
        *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(4)]
    )
    coordinate_decoder = Transformer_Decoder(
        tf_blocks,
        decoder_dim,
        cls_to_coords_res**2 + 1, # classifier
        is_classifier=True,
        amp=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True
    partial_conv_refiner = partial(
        ConvRefiner,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        corr_in_other=True,
        amp=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
        use_custom_corr=use_custom_corr,
    )

    conv_refiner = nn.ModuleDict({
        "8": partial_conv_refiner(
            2 * 320 + 64 + (2 * 3 + 1)**2,
            2 * 320 + 64 + (2 * 3 + 1)**2,
            2 + 1,
            displacement_emb_dim=64,
            local_corr_radius=3,
        ),
        "4": partial_conv_refiner(
            2 * 256 + 32 + (2 * 2 + 1)**2,
            2 * 256 + 32 + (2 * 2 + 1)**2,
            2 + 1,
            displacement_emb_dim=32,
            local_corr_radius=2,
        ),
        "2": partial_conv_refiner(
            2 * 64 + 16,
            128 + 16,
            2 + 1,
            displacement_emb_dim=16,
        ),
        "1": partial_conv_refiner(
            2 * 9 + 6,
            24,
            2 + 1,
            displacement_emb_dim=6,
        ),
    })
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp8 = GP(
        kernel=kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov
    )
    gps = nn.ModuleDict({"8": gp8})
    proj8 = nn.Sequential(nn.Conv2d(320, 320, 1, 1), nn.BatchNorm2d(320))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
    })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        embedding_decoder=coordinate_decoder,
        gps=gps,
        proj=proj,
        conv_refiner=conv_refiner,
        detach=True,
        scales=["8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )

    encoder = Coarse_Fine_Encoder(
        vgg_kwargs=dict(pretrained=encoder_pretrained, amp=True),
        repvgg_kwargs=dict(pretrained=encoder_pretrained, amp=True)
    )

    h, w = resolution
    matcher = RegressionMatcher(
        encoder=encoder,
        decoder=decoder,
        h=h,
        w=w,
        upsample_preds=upsample_preds,
        upsample_res=upsample_res,
        attenuate_cert=attenuate_cert,
        sample_mode=sample_mode,
        sample_thresh=sample_thresh,
        **kwargs,
    ).to(device)

    if not encoder_pretrained and tgm_weight is not None:
        matcher.load_state_dict(tgm_weight)
    return matcher


# if __name__ == "__main__":
#     from TGMatcher.utils.exp_utils import read_2_images
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = tgm_model(
#         resolution=256,
#         upsample_preds=True,
#         device=device,
#         use_custom_corr=True,
#         upsample_res=400
#     )
#     dir_path = r"E:\.Project\#Cross_Modal\TerrainGravityMatcher\data\gravity\2048_patch_256"
#     im1_path, im2_path = read_2_images(
#         im_A_path=os.path.join(dir_path, "patch_0.png"),
#         im_B_path=os.path.join(dir_path, "patch_1.png")
#     )
    
#     with torch.no_grad():
#         warp, certainty = model.match(im1_path, im2_path, device=device)
#     print(f"warp shape is {warp.shape}, certainty shape is {certainty.shape}")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     model_size_MB = total_params * 2 / (1024 * 1024)
#     print(f"Total params: {total_params}, Trainable params: {trainable_params}, Model size: {model_size_MB:.2f} MB")


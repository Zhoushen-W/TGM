import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
from PIL import Image
from einops import rearrange
from warnings import warn

from TGMatcher.model.encoder import Coarse_Fine_Encoder
from TGMatcher.utils.utils import (
    get_autocast_params, 
    cls_to_flow_refine,
    check_not_i16,
    check_rgb,
    get_tuple_transform_ops
)
from TGMatcher.utils.local_correlation import local_correlation
from TGMatcher.utils.kde import kde


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb=None,
        displacement_emb_dim=None,
        local_corr_radius=None,
        corr_in_other=None,
        no_im_B_fm=False,
        amp=False,
        concat_logits=False,
        use_bias_block_1=True,
        use_cosine_corr=False,
        disable_local_corr_grad=False,
        is_classifier=False,
        sample_mode="bilinear",
        norm_type=nn.BatchNorm2d,
        bn_momentum=0.1,
        amp_dtype=torch.float16,
        use_custom_corr=False,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim=in_dim,
            out_dim=hidden_dim,
            dw=dw,
            kernel_size=kernel_size,
            bias=use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(*[
            self.create_block(
                hidden_dim,
                hidden_dim,
                dw=dw,
                kernel_size=kernel_size,
                norm_type=norm_type,
            )
            for hb in range(hidden_blocks)
        ])
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emp = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        self.use_custom_corr = use_custom_corr

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, (
                "outdim must be divisible by indim for depthwise"
            )
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        if norm_type is nn.BatchNorm2d:
            norm = norm_type(out_dim, momentum=self.bn_momentum)
        else:
            norm = norm_type(num_channels=out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
    
    def forward(self, x, y, warp, scale_factor=1, logits=None):
        b, c, hs, ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            x_hat = F.grid_sample(
                input=y,
                grid=warp.permute(0, 2, 3, 1),
                align_corners=False,
                mode=self.sample_mode
            )
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                    (
                        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                    ),
                    indexing="ij",
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0])) # xy -> hw
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = warp - im_A_coords
                experience_param = 40 / 32 * scale_factor * in_displacement # 经验参数?
                emb_in_displacement = self.disp_emp(experience_param)
                if self.local_corr_radius:
                    if self.corr_in_other:
                        # Corr in other means take a kxk grid around the predicted coordinate in other image
                        local_corr = local_correlation(
                            x,
                            y,
                            self.local_corr_radius,
                            warp,
                            sample_mode=self.sample_mode,
                            use_custom_corr=self.use_custom_corr
                        )
                    else:
                        raise NotImplementedError(
                            "Local corr in own frame should not be used."
                        )
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty
    

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features=False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1),
                torch.arange(-(K // 2), K // 2 + 1),
                indexing="ij",
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2),
                torch.arange(K // 2, w + K // 2),
                indexing="ij",
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov
    
    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")
    
    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )
        
    def get_pos_env(self, y):
        b, c, h, w = y.shape
        coarse_coods = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing="ij",
        )
        coarse_coods = torch.stack(
            (
                coarse_coods[1],
                coarse_coods[0]
            ),
            dim=-1
        )[None].expand(b, h, w, 2)
        coarse_coods = rearrange(coarse_coods, "b h w d -> b d h w")
        coarse_embedded_coods = self.project_to_basis(coarse_coods)
        return coarse_embedded_coods
    
    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        # _, _, h2, w2 = y.shape
        f = self.get_pos_env(y)
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        # K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        if self.training:
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)
            mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        else:
            L_t = torch.linalg.cholesky(K_yy + sigma_noise)
            pos_emb = torch.cholesky_solve(f.reshape(b, h2 * w2, d), L_t, upper=False)
            mu_x = K_xy @ pos_emb
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1) # gp_feats
        return mu_x
    

class Decoder(nn.Module):
    """
    GP + Transformer + Refiner
    """
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        scales,
        detach=False,
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        self.pos_embeddings = pos_embeddings if pos_embeddings is not None else {}
        self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4 # why?
        self.dispacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing="ij",
        )
        coarse_coords = torch.stack(
            (
                coarse_coords[1],
                coarse_coords[0]
            ),
            dim=-1
        )[None].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    # def get_positional_embedding(self, b, h, w, device):
    #     coarse_coords = torch.meshgrid(
    #         (
    #             torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
    #             torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
    #         ),
    #         indexing="ij",
    #     )
    #     coarse_coords = torch.stack(
    #         (
    #             coarse_coords[1],
    #             coarse_coords[0]
    #         ),
    #         dim=-1
    #     )[None].expand(b, h, w, 2)
    #     coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
    #     coarse_embedded_coords = self.pos_embedding(coarse_coords)
    #     return coarse_embedded_coords
    
    def forward(
        self, 
        f1,
        f2,
        gt_warp=None,
        gt_prob=None,
        upsample=False,
        flow=None,
        certainty=None,
        scale_factor=1,
    ):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["4", "2", "1"] # fine_features
        sizes = {scale: f1[scale].shape[-2:] for scale in f1} # let scales=["8", "4", "2", "1"] when use Decoder
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b,
            self.embedding_decoder.hidden_dim,
            *sizes[coarsest_scale],
            device=f1[coarsest_scale].device,
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device=device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                flow,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            certainty = F.interpolate(
                certainty,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(
                    f1_s.device, 
                    str(f1_s) == "cuda", 
                    self.amp_dtype
                )
                with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff,
                    size=sizes[ins],
                    mode="bilinear",
                    align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder( 
                    gp_posterior,
                    f1_s,
                    old_stuff,
                    new_scale
                )
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(gm_warp_or_cls).permute(0, 3, 1, 2)
                    corresps[ins].update({
                        "gm_cls": gm_warp_or_cls,
                        "gm_certainty": certainty,
                    }) if self.training else None
                else:
                    corresps[ins].update({
                        "gm_flow": gm_warp_or_cls,
                        "gm_certainty": certainty,
                    }) if self.training else None
                    flow = gm_warp_or_cls.detach()

            if new_scale in self.conv_refiner:
                corresps[ins].update({
                    "flow_pre_delta": flow
                }) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s,
                    f2_s,
                    flow,
                    scale_factor=scale_factor,
                    logits=certainty,
                )
                corresps[ins].update({
                    "delta_flow": delta_flow
                }) if self.training else None
                displacement = ins * torch.stack(
                    (
                        delta_flow[:, 0].float() / (self.refine_init * w),
                        delta_flow[:, 1].float() / (self.refine_init * h),
                    ),
                    dim=1,
                )
                flow = flow + displacement
                certainty = certainty + delta_certainty
            
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
        return corresps
        

def _check_input(im_input):
    if isinstance(im_input, (str, os.PathLike)):
        im = Image.open(im_input)
        check_not_i16(im)
        im = im.convert("RGB")
    elif isinstance(im_input, Image.Image):
        check_rgb(im_input)
        im = im_input
    else:
        assert isinstance(im_input, torch.Tensor), (
            "im_input must be a string, path, or PIL image"
        )
        B, C, H, W = im_input.shape
        assert C == 3, "im_input must be a 3channel image"
        # assert H % 14 == 0, "im_input must be a multiple of 14"
        # assert W % 14 == 0, "im_input must be a multiple of 14"
        im = im_input
    return im


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder: Coarse_Fine_Encoder,
        decoder: Decoder,
        h=256, # Will this be valid?
        w=256,
        sample_mode="threshold_balanced",
        upsample_preds=False,
        symmetric=False,
        sample_thresh=0.05,
        name=None,
        attenuate_cert=None,
        upsample_res=None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res or (16 * 8 * 6, 16 * 8 * 6) # will this be valid?
        self.symmetric = symmetric
        self.sample_thresh = sample_thresh

    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
        
    def extract_backbone_features(self, batch, batched=True):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        # print(f"x_q: {x_q.shape}, x_s: {x_s.shape}")
        if batched:
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X)
        else:
            feature_pyramid = (
                self.encoder(x_q),
                self.encoder(x_s),
            )
        return feature_pyramid
    
    def sample(
        self, 
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * num, len(certainty)),
            replacement=False,
        )
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7
        balanced_samples = torch.multinomial(
            p,
            num_samples=min(num, len(good_certainty)),
            replacement=False,
        )
        return good_matches[balanced_samples], good_certainty[balanced_samples]
    
    def forward(self, batch, batched=True, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(batch=batch, batched=batched)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(
            f1=f_q_pyramid,
            f2=f_s_pyramid,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor
        )
        return corresps

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B=None, W_B=None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A)
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)
        
    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack(
            (W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)),
            axis=-1
        )
        return kpts
    
    def _get_device(self):
        return self.encoder.vgg.layers[0].weight.device
    
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        im_A_high_res=None,
        im_B_high_res=None,
        batched=True,
        device=None,
    ):
        self.train(False)
        if not batched:
            raise ValueError(
                "batched must be True, non-batched inference is no longer supported."
            )
        if device is None and not isinstance(im_A_input, torch.Tensor):
            device = self._get_device()
        elif device is None and isinstance(im_A_input, torch.Tensor):
            device = im_A_input.device

        im_A, im_B = _check_input(im_A_input), _check_input(im_B_input)
        ws = self.w_resized
        hs = self.h_resized
        scale_factor = math.sqrt(hs * ws / (256**2)) # 假设采样到256
        if isinstance(im_A, Image.Image) and isinstance(im_B, Image.Image):
            b = 1
            w1, h1 = im_A.size
            w2, h2 = im_B.size
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws),
                normalize=True,
                clahe=False
            )
            im_A, im_B = test_transform((im_A, im_B))
            batch = {
                "im_A": im_A[None].to(device),
                "im_B": im_B[None].to(device)
            }
        elif isinstance(im_A, torch.Tensor) and isinstance(im_B, torch.Tensor):
            b, c, h1, w1 = im_A.shape
            b, c, h2, w2 = im_B.shape
            assert w1 == w2 and h1 == h2, "For batched images we assume same size"
            batch = {
                "im_A": im_A.to(device),
                "im_B": im_B.to(device)
            }
            if h1 != self.h_resized or self.w_resized != w1:
                warn(
                    "Model resolution and batch resolution differ, may produce unexpected results"
                )
            hs, ws = h1, w1
        else:
            raise ValueError(
                f"Unsupported input type: {type(im_A)=} and {type(im_B)=}"
            )
        finest_scale = 1

        # run matcher
        corresps = self.forward(batch, batched=True, scale_factor=scale_factor)
        if self.upsample_preds:
            hs, ws = self.upsample_res
        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
                corresps[8]["certainty"],
                size=(hs, ws),
                align_corners=False,
                mode="bilinear",
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)
        finest_corresps = corresps[finest_scale]

        if self.upsample_preds and im_A_high_res is None and im_B_high_res is None:
            torch.cuda.empty_cache()
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws),
                normalize=True,
            )
            if isinstance(im_A_input, (str, os.PathLike)):
                im_A, im_B = test_transform(
                    (
                        Image.open(im_A_input).convert("RGB"),
                        Image.open(im_B_input).convert("RGB"),
                    )
                )
            else:
                assert isinstance(im_A_input, Image.Image), f"Unsupported input type: {type(im_A_input)=}"
                assert isinstance(im_B_input, Image.Image), f"Unsupported input type: {type(im_B_input)=}"
                im_A, im_B = test_transform((im_A_input, im_B_input))
            im_A, im_B = im_A[None].to(device), im_B[None].to(device)
            batch = {
                "im_A": im_A,
                "im_B": im_B,
                "corresps": finest_corresps
            }
        elif self.upsample_preds and im_A_high_res is not None and im_B_high_res is not None:
            batch = {
                "im_A": im_A_high_res,
                "im_B": im_B_high_res,
                "corresps": finest_corresps
            }
        elif self.upsample_preds:
            raise ValueError(
                f"Invalid upsample_preds and high_res inputs with {im_A=},{im_A_high_res=},{im_B=} and {im_B_high_res=}"
            )
        if self.upsample_preds:
            scale_factor = math.sqrt(
                self.upsample_res[0] * self.upsample_res[1] / (256**2)
            )
            corresps = self.forward(batch, batched=True, scale_factor=scale_factor)
        
        im_A_to_im_B = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"] - (
            low_res_certainty if self.attenuate_cert else 0
        )
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B,
                size=(hs, ws),
                align_corners=False,
                mode="bilinear",
            )
            certainty = F.interpolate(
                certainty,
                size=(hs, ws),
                align_corners=False,
                mode="bilinear",
            )
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            ),
            indexing="ij",
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        certainty = certainty.sigmoid() # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        if batched:
            return (warp, certainty[:, 0])
        else:
            return (warp[0], certainty[0, 0])
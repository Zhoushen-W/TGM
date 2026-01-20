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

from TGMatcher.utils.utils import get_grid, get_autocast_params


class Transformer_Decoder(nn.Module):
    def __init__(
        self, 
        blocks,
        hidden_dim,
        out_dim,
        is_classifier=False,
        amp=False,
        amp_dtype=torch.float16,
        pos_enc=True,
        learned_embeddings=False,
        embedding_dim=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [8] # 1/8 Features
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            self.learned_pos_embeddings = nn.Parameter(
                nn.init.kaiming_normal_(
                    torch.empty(1, hidden_dim, embedding_dim, embedding_dim)
                )
            )

    def scales(self):
        return self._scales.copy()
    
    def forward(self, gp_posterior, features, out_stuff, new_scale):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(gp_posterior.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            b, c, h, w = gp_posterior.shape
            x = torch.cat((gp_posterior, features), dim=1)
            b, c, h, w = x.shape
            # grid = get_grid(b, h, w, x.device).reshape(b, h * w, 2)
            if self.learned_embeddings:
                pos_enc = F.interpolate(
                    self.learned_pos_embeddings,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                pos_enc = pos_enc.permute(0, 2, 3, 1).reshape(1, h * w, c)
            else:
                pos_enc = 0
            tokens = x.reshape(b, c, h * w).permute(0, 2, 1) + pos_enc
            z = self.blocks(tokens)
            out = self.to_out(z)
            out = out.permute(0, 2, 1).reshape(b, self.out_dim, h, w)
            warp, certainty = out[:, :-1], out[:, -1:] # classifier
            # warp, certainty = out[:, :2], out[:, 2:3]
            return warp , certainty, None 


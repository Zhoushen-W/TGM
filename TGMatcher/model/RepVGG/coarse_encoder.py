import torch
import torch.nn as nn

from TGMatcher.model.RepVGG.repvgg import create_RepVGG_B2
from TGMatcher.utils.utils import get_autocast_params


class RepVGG_8(nn.Module):
    """
    Get 1/8 features from RepVGG

    Pretrained weights is in path
    """
    def __init__(self, pretrained=True, amp=False, amp_dtype=torch.float16):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        backbone = create_RepVGG_B2(deploy=False) # train_mode: deploy=False
        if pretrained:
            path = "./TGMatcher/model/RepVGG/weights/RepVGG-B2-train.pth"
            weights = torch.load(f=path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(weights)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            x = self.layer0(x) # 1/2
            for module in self.layer1:
                x = module(x) # 1/4
            for module in self.layer2:
                x = module(x) # 1/8
            # for module in self.layer3:
            #     x = module(x) # 1/16
            return x
import torch
import torch.nn as nn
import torchvision.models as tvm

from TGMatcher.utils.utils import get_autocast_params
from TGMatcher.model.RepVGG.coarse_encoder import RepVGG_8


class VGG19(nn.Module):
    def __init__(self, pretrained=True, amp=False, amp_type=torch.float16):
        super().__init__()
        if pretrained:
            self.weights = tvm.VGG19_BN_Weights.IMAGENET1K_V1
        else:
            self.weights = None
        self.layers = nn.ModuleList(tvm.vgg19_bn(weights=self.weights).features[:27]) # only to 1/8 features
        self.amp = amp
        self.amp_dtype = amp_type

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            feats = {}
            scale = 1
            for layers in self.layers:
                if isinstance(layers, nn.MaxPool2d):
                    feats[scale] = x
                    scale *= 2
                x = layers(x)
            return feats
        

class Coarse_Fine_Encoder(nn.Module):
    """
    When training, pretrained could be True;
    
    When evaluating, pretrained should be False.
    """
    def __init__(self, vgg_kwargs=None, repvgg_kwargs=None):
        super().__init__()
        self.vgg = VGG19(**vgg_kwargs)
        self.repvgg = RepVGG_8(**repvgg_kwargs)

    def train_vgg(self, if_train=True):
        return self.vgg.train(if_train)
    
    def train_repvgg(self, if_train=True):
        return self.repvgg.train(if_train)
    
    def forward(self, x):
        # B, C, H, W = x.shape
        feature_pyramid = self.vgg(x)
        feature_8 = self.repvgg(x)
        feature_pyramid[8] = feature_8
        return feature_pyramid
    
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Coarse_Fine_Encoder(
#         vgg_kwargs={"pretrained":True, "amp":True},
#         repvgg_kwargs={"pretrained":True, "amp":True}
#     ).to(device)
#     x = torch.randn(1, 3, 256, 256).to(device)
#     with torch.no_grad():
#         feats = model(x)
#     for scale, f in feats.items():
#         print(f"Scale 1/{scale}: shape = {tuple(f.shape)}")



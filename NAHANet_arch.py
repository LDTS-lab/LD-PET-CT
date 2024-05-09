import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from codes.models.modules.hat_arch import HAT

class NAHANet(HAT):


    def forward(self, x):
        fea_L = self.conv_first(x[0])
        fea_noise = self.Convnoise(x[1])
        fea = (fea_L,fea_noise)
        x = self.forward_features(fea)
        x = self.conv_before_upsample(x)
        x = self.conv_last(x)
        return x

if __name__ == "__main__":

    model = NAHANet(img_size=64,patch_size=4,in_chans=1,
                                          embed_dim=36,depths=(2,2,6,2),window_size=4)

    x = (torch.randn((2, 1, 64, 64)),torch.randn((2, 1, 64, 64)))
    x = model(x)
    print(x.shape)
    from thop import profile

    data = (torch.randn((2, 1, 64, 64)),torch.randn((2, 1, 64, 64)))
    out = model(data)
    flops, params = profile(model, inputs=(data,))
    print(flops, params)

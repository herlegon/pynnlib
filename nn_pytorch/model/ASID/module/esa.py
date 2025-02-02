import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels: int, n_feats: int, conv=nn.Conv2d):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


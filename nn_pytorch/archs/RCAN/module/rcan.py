import torch.nn as nn
from torch import Tensor

from .common import (
    MeanShift,
    Upsampler,
    default_conv,
)
from ..._shared.pad import pad, unpad



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1
    ):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res



## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super().__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        res += x
        return res



class RCAN(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        reduction: int = 16,
        rgb_range: int = 255,
        norm: bool = True,
        n_colors: int = 3,
        kernel_size: int = 3,
        res_scale: int = 1,
        conv=default_conv
    ):
        super().__init__()

        self.scale: int = scale

        if norm:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_std = (1.0, 1.0, 1.0)
            self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        else:
            self.rgb_range = 1
            self.sub_mean = nn.Identity()

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        act = nn.ReLU(True)
        modules_body = [
            ResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act=act,
                res_scale=res_scale,
                n_resblocks=n_resblocks
            )
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        if norm:
            self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        else:
            self.add_mean = nn.Identity()

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, x: Tensor) -> Tensor:
        # NCHW
        size = x.shape[2:]
        x = pad(x, modulo=2, mode='reflect')

        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        out = unpad(out, size, scale=self.scale)

        x /= self.rgb_range
        return x

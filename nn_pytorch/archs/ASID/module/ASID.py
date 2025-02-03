import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ids import (
    IDSG_A,
    IDSG,
)


def pixelshuffle_block(
    in_nc: int,
    out_nc: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    bias: bool = False
):
    """
    Upsample features according to `upscale_factor`.
    """
    # padding = kernel_size // 2
    conv = nn.Conv2d(
        in_nc,
        out_nc * (upscale_factor ** 2),
        kernel_size,
        padding=1,
        bias=bias
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])



class ASID(nn.Module):
    def __init__(
        self,
        in_nc: int,
        out_nc: int,
        scale: int,
        window_size: int = 8,
        num_feat: int = 64,
        res_num: int = 8,
        block_num: int = 1,
        bias: bool = True,
        pe: bool = False,
        d8: bool = False
    ):
        super().__init__()

        self.res_num: int = res_num
        self.window_size: int = window_size
        self.scale: int = scale
        self.d8: bool = d8

        self.block0 = IDSG_A(
            channel_num=num_feat,
            bias=bias,
            block_num=block_num,
            window_size=window_size,
            pe=pe
        )

        self.block1 = IDSG(
            channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
        )
        self.block2 = IDSG(
            channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
        )
        if d8:
            self.block3 = IDSG(
                channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
            )
            self.block4 = IDSG(
                channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
            )
            self.block5 = IDSG(
                channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
            )
            self.block6 = IDSG(
                channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
            )
            self.block7 = IDSG(
                channel_num=num_feat, bias=bias, window_size=window_size, pe=pe
            )

        self.input  = nn.Conv2d(
            in_channels=in_nc,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.up = pixelshuffle_block(
            in_nc=num_feat,
            out_nc=out_nc,
            upscale_factor=scale,
            bias=bias
        )


    def check_image_size(self, x: Tensor) -> Tensor:
        h, w = x.size()[2:]
        modulo: int = 32
        mod_pad_h = (modulo - h % modulo) % modulo
        mod_pad_w = (modulo - w % modulo) % modulo
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x


    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out, a1, a2, a3, a4, a5, a6 = self.block0(residual)
        out = self.block1(out, a1, a2, a3, a4, a5, a6)
        out = self.block2(out, a1, a2, a3, a4, a5, a6)
        if self.d8:
            out = self.block3(out, a1, a2, a3, a4, a5, a6)
            out = self.block4(out, a1, a2, a3, a4, a5, a6)
            out = self.block5(out, a1, a2, a3, a4, a5, a6)
            out = self.block6(out, a1, a2, a3, a4, a5, a6)
            out = self.block7(out, a1, a2, a3, a4, a5, a6)

        # origin
        out = torch.add(self.output(out),residual)
        out = self.up(out)

        out = out[:, :, : h * self.scale, : w * self.scale]
        return  out

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from . import block as B
from ..._shared.pad import pad, unpad


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        num_filters: int = 64,
        num_blocks: int = 23,
        scale: int = 4,
        c2x2: bool = False,
        plus: bool = False,
        shuffle_factor: int | None = None,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super().__init__()

        self.shuffle_factor = shuffle_factor
        self.scale = scale
        if self.shuffle_factor is not None:
            in_nc = in_nc * self.shuffle_factor**2
            self.scale = self.scale * self.shuffle_factor

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{upsampler}] is not found")

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=num_filters,
                out_nc=num_filters,
                upscale_factor=3,
                act_type=act,
                c2x2=c2x2,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=num_filters,
                    out_nc=num_filters,
                    act_type=act,
                    c2x2=c2x2,
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=in_nc,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=norm,
                            act_type=act,
                            mode="CNA",
                            plus=plus,
                            c2x2=c2x2,
                        )
                        for _ in range(num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=num_filters,
                        out_nc=num_filters,
                        kernel_size=3,
                        norm_type=norm,
                        act_type=None,
                        mode=mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=num_filters,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=act,
                c2x2=c2x2,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=num_filters,
                out_nc=out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )


    def forward(self, x: Tensor) -> Tensor:
        # Assume the tensor is clamped before
        # AND the caller will clamp it after the inference
        x0 : Tensor = x

        if self.shuffle_factor:
            size = x.shape[2:]
            x0 = pad(x0, modulo=self.shuffle_factor, mode='reflect')

            x0 = torch.pixel_unshuffle(x0, downscale_factor=self.shuffle_factor)
            x0 = self.model(x0)

            out = unpad(x0, size, scale=self.scale)

        else:
            out = self.model(x0)

        return out

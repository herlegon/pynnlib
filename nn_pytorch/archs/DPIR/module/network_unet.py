from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from ..._shared.basicblock import (
    PixelUnShuffle,
    conv,
    downsample_maxpool,
    downsample_avgpool,
    downsample_strideconv,
    IMDBlock,
    NonLocalBlock2D,
    ResBlock,
    sequential,
    upsample_upconv,
    upsample_convtranspose,
    upsample_upconv,
    upsample_pixelshuffle,
)



class UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNet, self).__init__()

        self.m_head = conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = sequential(*[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = sequential(*[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = sequential(*[conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0


        return x



class UNetRes(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode='strideconv',
        upsample_mode='convtranspose'
    ):
        super(UNetRes, self).__init__()
        in_nc += 1

        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')


    def forward(self, x0):
        w = x0.shape[3] if x0.shape[1] <= 3 else x0.shape[1]
        h = x0.shape[2]

        mod_pad_h: int = (8 - h % 8) % 8
        mod_pad_w: int = (8 - w % 8) % 8
        x0 = F.pad(x0, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        _h, _w = x0.shape[2:]
        noise_level_model: float = 7./255.
        x0 = torch.cat(
            (
                x0,
                torch.full((1, 1, _h, _w), noise_level_model, dtype=x0.dtype, device=x0.device)
            ),
            dim=1
        )

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[:, :, :h, :w]
        return x



class ResUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        out_nc: int = 1,
        nc: tuple[int] = [64, 128, 256, 512],
        nb: int = 4,
        act_mode: str = "L",
        downsample_mode: Literal['avgpool', 'maxpool', 'strideconv'] = 'strideconv',
        upsample_mode: Literal['upconv', 'pixelshuffle', 'convtranspose'] = 'convtranspose'
    ):
        super(ResUNet, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = sequential(*[IMDBlock(nc[3], nc[3], bias=False, mode='C'+act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        h, w = x.size()[-2:]
        modulo: int = 8
        mod_pad_h: int = (modulo - h % modulo) % modulo
        mod_pad_w: int = (modulo - w % modulo) % modulo
        x = nn.ReplicationPad2d((0, mod_pad_w, 0, mod_pad_h))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = x[..., :h, :w]

        return x



class UNetResSubP(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetResSubP, self).__init__()
        sf = 2
        self.m_ps_down = PixelUnShuffle(sf)
        self.m_ps_up = nn.PixelShuffle(sf)
        self.m_head = conv(in_nc*sf*sf, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = sequential(*[ResBlock(nc[3], nc[3], mode='C'+act_mode+'C') for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc*sf*sf, bias=False, mode='C')

    def forward(self, x0):
        x0_d = self.m_ps_down(x0)
        x1 = self.m_head(x0_d)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = self.m_ps_up(x) + x0

        return x



class UNetPlus(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetPlus, self).__init__()

        self.m_head = conv(in_nc, nc[0], mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode[1]))
        self.m_down2 = sequential(*[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode[1]))
        self.m_down3 = sequential(*[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode[1]))

        self.m_body  = sequential(*[conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb-1)], conv(nc[2], nc[2], mode='C'+act_mode[1]))
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb-1)], conv(nc[1], nc[1], mode='C'+act_mode[1]))
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb-1)], conv(nc[0], nc[0], mode='C'+act_mode[1]))

        self.m_tail = conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x



class NonLocalUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: list[int, int, int, int] = [64, 128, 256, 512],
        nb: int = 1,
        act_mode: str = 'R',
        downsample_mode: Literal['avgpool', 'maxpool', 'strideconv'] = 'strideconv',
        upsample_mode: Literal['upconv', 'pixelshuffle', 'convtranspose'] = 'convtranspose'
    ):
        super(NonLocalUNet, self).__init__()

        down_nonlocal = NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')
        up_nonlocal = NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')

        self.m_head = conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        self.m_down1 = sequential(*[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = sequential(*[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = sequential(down_nonlocal, *[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = sequential(*[conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.m_up3 = sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, mode='C')

    def forward(self, x0: Tensor) -> Tensor:
        w = x0.shape[3] if x0.shape[1] <= 3 else x0.shape[1]
        h = x0.shape[2]

        mod_pad_h: int = (16 - h % 16) % 16
        mod_pad_w: int = (16 - w % 16) % 16
        x0 = F.pad(x0, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0

        x = x[:, :, :h, :w]
        return x


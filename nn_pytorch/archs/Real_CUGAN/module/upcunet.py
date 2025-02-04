# https://github.com/bilibili/ailab/blob/main/Real-CUGAN/upcunet_v3.py

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
from ..._shared.pad import pad, unpad


class SEBlock(nn.Module):
    def __init__(
        self,
        in_nc:int,
        reduction: int=8,
        bias: bool=False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_nc, in_nc // reduction, 1, 1, 0, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_nc // reduction, in_nc, 1, 1, 0, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        if "Half" in x.type():
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x: Tensor, x0: Tensor) -> Tensor:
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x



class UNetConv(nn.Module):
    def __init__(self, in_nc: int, mid_nc: int, out_nc: int, se: bool):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_nc, out_nc, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_nc, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z



class UNet1(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, deconv: bool):
        super().__init__()
        self.conv1 = UNetConv(in_nc, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_nc, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_nc, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)

        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z


    def forward_a(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2


    def forward_b(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z



class UNet1x3(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, deconv: bool):
        super().__init__()
        self.conv1 = UNetConv(in_nc, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_nc, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_nc, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z



class UNet2(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, deconv: bool):
        super().__init__()

        self.conv1 = UNetConv(in_nc, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_nc, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_nc, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, alpha: float) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # conv234 ends with se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2: Tensor) -> Tensor:
        # conv234 ends with se
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2: Tensor, x3: Tensor) -> Tensor:
        # conv234 ends with se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1: Tensor, x4: Tensor) -> Tensor:
        # conv234 ends with se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z



class UpCunet(nn.Module):
    def __init__(self,
        in_nc: int = 3,
        out_nc: int = 3,
        scale: int = 2,
        alpha: float = 1.0,
        shuffle_factor: int = 0,
        legacy: bool = False,
    ):
        super().__init__()

        self.modulo = 2
        if scale == 2:
            self.unet1 = UNet1(in_nc, out_nc, deconv=True)
            self.unet2 = UNet2(in_nc, out_nc, deconv=False)
            self.pad = 18
        elif scale == 3:
            self.unet1 = UNet1x3(in_nc, out_nc, deconv=True)
            self.unet2 = UNet2(in_nc, out_nc, deconv=False)
            self.pad = 14
            self.modulo = 4
        elif scale == 4:
            self.unet1 = UNet1(in_nc, 64, deconv=True)
            self.unet2 = UNet2(64, 64, deconv=False)
            self.ps = nn.PixelShuffle(shuffle_factor)
            self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)
            self.pad = 19
        else:
            raise ValueError(f"Scale {scale} is not supported")

        self.scale = scale
        self.legacy = legacy
        self.alpha = alpha if 0.75 <= alpha <= 1.3 else 1.


    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]
        x = torch.clamp_(x, 0, 1)

        if not self.legacy:
            # Should not be inplace operations if requires_grad=True
            x.mul_(0.7).add_(0.15)
        _x: Tensor = x

        x = pad(x, modulo=self.modulo, mode='reflect')

        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x, self.alpha)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)

        if self.scale == 4:
            x = self.conv_final(x)
            x = F.pad(x, (-1, -1, -1, -1))
            x = self.ps(x)

        x = unpad(x, size, scale=self.scale)

        if self.scale == 4:
            x += F.interpolate(_x, scale_factor=self.scale, mode="nearest")

        if not self.legacy:
            # Should not be inplace operations if requires_grad=True
            x.sub_(0.15).div_(0.7)

        return x


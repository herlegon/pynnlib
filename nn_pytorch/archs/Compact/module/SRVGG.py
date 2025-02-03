# https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
# modified
from typing import Literal
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        in_nc (int): Channel number of inputs. Default: 3.
        out_nc (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self,
            in_nc: int=3,
            out_nc: int=3,
            num_feat: int=64,
            num_conv: int=16,
            scale: int=4,
            act_type: Literal['relu', 'prelu', 'leakyrelu']='prelu',
        ):
        super(SRVGGNetCompact, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.scale = scale
        self.act_type = act_type

        self.body: nn.ModuleList = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(in_nc, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, out_nc * scale * scale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(scale)


    def forward(self, x: Tensor) -> Tensor:
        # Assume the tensor is clamped before
        # AND the caller will clamp it after the inference
        out: Tensor = x
        for body in self.body:
            out = body(out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        out += F.interpolate(x, scale_factor=self.scale, mode='nearest')

        return out

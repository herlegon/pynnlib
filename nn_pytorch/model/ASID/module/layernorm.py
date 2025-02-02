#############################################################
# File: layernorm.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:28:20 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import torch
import torch.nn as nn
from torch import Tensor


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, weight, bias, eps) -> Tensor:
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None
        )



class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps


    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

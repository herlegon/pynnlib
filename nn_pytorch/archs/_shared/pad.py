from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor


def pad(
    x: torch.Tensor,
    modulo: int,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0
) -> Tensor:
    h, w = x.shape[2:]
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode, value=value)

    return x

def unpad(x: Tensor, size: tuple[int, int], scale: int = 1) -> Tensor:
    # size: (h, w)
    h, w = size
    return x[:, :, :h * scale, :w * scale]

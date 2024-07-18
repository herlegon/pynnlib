from __future__ import annotations
from collections.abc import Callable
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
from torch import Tensor
from pynnlib import is_cuda_available
from pynnlib.logger import nnlogger
from pynnlib.model import PytorchModel
from pynnlib.architecture import SizeConstraint
from pynnlib.session import GenericSession
from pynnlib.utils.torch_tensor import (
    to_nchw,
    flip_r_b_channels,
    to_hwc
)
from ..torch_types import TorchNnModule


class PyTorchSession(GenericSession):
    """An example of session used to perform the inference using a PyTorch model.
    There is a reason why the initialization is not done in the __init__: mt/mp
    """

    def __init__(self, model: PytorchModel):
        super().__init__()
        self.model = model
        self.device: torch.device = torch.device('cpu')

        self.model.module.load_state_dict(self.model.state_dict)
        for _, v in self.model.module.named_parameters():
            v.requires_grad = False

        self._run_fct: Callable[[np.ndarray], np.ndarray | None] = self._run_torch


    @property
    def module(self) -> TorchNnModule:
        return self.model.module


    def initialize(
        self,
        device: str,
        fp16: bool = False,
        warmup: bool = True,
    ) -> None:
        module: nn.Module = self.module
        self.fp16 = fp16
        self.device = device
        if not is_cuda_available():
            self.fp16 = False
            self.device: str = 'cpu'
        self.fp16 = self.fp16 and 'fp16' in self.model.arch.dtypes

        nnlogger.debug(f"[V] Initialize a PyTorch inference session fp16={self.fp16}")
        torch.backends.cudnn.benchmark = True
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

        nnlogger.debug(f"[V] load model to {self.device}, fp16={self.fp16}")
        module.to(self.device)
        module.half() if self.fp16 else module.float()
        if warmup and 'cuda' in device:
            self.warmup(3)


    def warmup(self, count: int = 1):
        size_constraint: SizeConstraint = self.model.size_constraint
        if size_constraint is not None and size_constraint.min is not None:
            shape = (*reversed(size_constraint.min), self.model.in_nc)
        else:
            shape = (32, 32, self.model.in_nc)
        nnlogger.debug(f"[V] warmup ({count}x) with a random img ({shape})")
        img = np.random.random(shape).astype(np.float32)
        for _ in range(count):
            self._run_torch(img)


    def run(self, in_img: np.ndarray) -> np.ndarray:
        return self._run_fct(in_img)


    @torch.inference_mode()
    def _run_torch(self, in_img: np.ndarray) -> np.ndarray:
        in_tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(self.device, dtype=torch.float32)
        in_tensor = in_tensor.half() if self.fp16 else in_tensor.float()
        in_tensor = flip_r_b_channels(in_tensor)
        in_tensor = to_nchw(in_tensor).contiguous()

        out_tensor: Tensor = self.module(in_tensor)
        out_tensor = torch.clamp_(out_tensor, 0, 1)

        out_tensor = to_hwc(out_tensor)
        out_tensor = flip_r_b_channels(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return out_img





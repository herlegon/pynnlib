from __future__ import annotations
from collections.abc import Callable
import time
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pynnlib.architecture import InferType


class PyTorchSession(GenericSession):
    """An example of session used to perform the inference using a PyTorch model.
    There is a reason why the initialization is not done in the __init__: mt/mp
    It maybe be overloaded by a customized session
    """

    def __init__(self, model: PytorchModel):
        super().__init__()
        self.model = model
        self.device: torch.device = torch.device('cpu')

        self.model.module.load_state_dict(self.model.state_dict)
        for _, v in self.model.module.named_parameters():
            v.requires_grad = False

        self._process_fct: Callable[[np.ndarray], np.ndarray] = self._torch_process
        infer_type: InferType = model.arch.infer_type
        if (
            infer_type.type == 'inpaint'
            and infer_type.inputs == 2
            and infer_type.outputs == 1
        ):
            self._process_fct = self._torch_process_inpaint

        elif (
            infer_type.type != 'simple'
            or infer_type.inputs != 1
            or infer_type.outputs != 1
        ):
            raise NotImplementedError(f"Cannot create a session for arch {model.arch_name} ")


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
        module = module.half() if self.fp16 else module.float()
        if warmup and 'cuda' in device:
            self.warmup(3)


    def warmup(self, count: int = 1):
        size_constraint: SizeConstraint = self.model.size_constraint
        if size_constraint is not None and size_constraint.min is not None:
            shape = (*reversed(size_constraint.min), self.model.in_nc)
        else:
            shape = (32, 32, self.model.in_nc)
        nnlogger.debug(f"[V] warmup ({count}x) with a random img ({shape})")

        # TODO: specific warmup
        imgs: list[np.ndarray] = list([
            np.random.random(shape).astype(np.float32)
            for _ in range(self.model.arch.infer_type.inputs)
        ])
        for _ in range(count):
            self._process_fct(*imgs)


    def process(self, in_img: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._process_fct(in_img, *args, **kwargs)


    @torch.inference_mode()
    def _torch_process(self, in_img: np.ndarray) -> np.ndarray:
        """Example of how to perform an inference session.
        This is an unoptimized function
        """
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


    @torch.inference_mode()
    def _torch_process_inpaint(self, in_img: np.ndarray, in_img2: np.ndarray) -> np.ndarray:
        # Used for inpaiting
        in_tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(self.device, dtype=torch.float32)
        in_tensor = in_tensor.half() if self.fp16 else in_tensor.float()
        in_tensor = flip_r_b_channels(in_tensor)
        in_tensor = to_nchw(in_tensor).contiguous()

        if len(in_img2.shape) > 2:
            in_img2 = in_img2[:,:,0]
        in_tensor2 = torch.from_numpy(np.ascontiguousarray(in_img2))
        in_tensor2 = in_tensor2.to(self.device, dtype=torch.float32)
        in_tensor2 = in_tensor2.half() if self.fp16 else in_tensor2.float()
        in_tensor2 = to_nchw(in_tensor2).contiguous()

        start_time = time.time()
        out_tensor: Tensor = self.module(in_tensor, in_tensor2)
        out_tensor = torch.clamp_(out_tensor, 0, 1)
        print(f"elapsed: {1000 * (time.time() - start_time):.1f}ms")

        out_tensor = to_hwc(out_tensor)
        out_tensor = flip_r_b_channels(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return out_img


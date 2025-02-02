from __future__ import annotations
from collections.abc import Callable
import math
from pprint import pprint
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import TYPE_CHECKING
from pynnlib import is_cuda_available
from pynnlib.logger import nnlogger
from pynnlib.model import PytorchModel
from pynnlib.architecture import SizeConstraint
from pynnlib.session import GenericSession
from pynnlib.utils.torch_tensor import (
    to_nchw_torch,
    flip_r_b_channels_torch,
    to_hwc_torch
)
from ..torch_types import TorchNnModule
if TYPE_CHECKING:
    from pynnlib.architecture import InferType
import cupy as cp
from torch import (
    from_dlpack,
    to_dlpack,
    Tensor,
)


class PyTorchSession(GenericSession):
    """An example of session used to perform the inference using a PyTorch model.
    There is a reason why the initialization is not done in the __init__: mt/mp
    It maybe be overloaded by a customized session
    """

    def __init__(self, model: PytorchModel):
        super().__init__()
        self.model = model
        self.device: torch.device = torch.device('cpu')

        self.model.module.load_state_dict(self.model.state_dict, strict=False)
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
        torch.backends.cudnn.enabled = True

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
        torch_transfer: bool = False
        if torch_transfer:
            in_tensor = torch.from_numpy(np.ascontiguousarray(in_img))
            h_tensor = torch.empty(in_tensor.shape, pin_memory=True)
        else:
            # HtoD
            # allocate 2 times the image size for testing
            cp_cuda_stream = cp.cuda.stream.Stream(non_blocking=True)
            with cp_cuda_stream:
                h_in_mem = cp.cuda.alloc_pinned_memory(
                    2 * (math.prod(in_img.shape) * np.dtype(np.float32).itemsize)
                )

                h_input = np.frombuffer(
                    h_in_mem,
                    dtype=in_img.dtype,
                    count=math.prod(in_img.shape)
                ).reshape(in_img.shape)
                h_input[...] = np.ascontiguousarray(in_img)
                d_cp_tensor = cp.empty(in_img.shape, in_img.dtype)
                d_cp_tensor.set(h_input)
                d_cp_tensor = d_cp_tensor.astype(in_img.dtype)
                d_tensor: Tensor = from_dlpack(d_cp_tensor.toDlpack())
            cp_cuda_stream.synchronize()


        infer_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(infer_stream):
            if torch_transfer:
                h_tensor.copy_(in_tensor).detach()
                d_tensor: Tensor = h_tensor.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                torch.cuda.synchronize()


            # in_tensor = in_tensor.to(self.device, dtype=torch.float32)
            in_tensor: Tensor = d_tensor.half() if self.fp16 else d_tensor.float()
            in_tensor = flip_r_b_channels_torch(in_tensor)
            in_tensor = to_nchw_torch(in_tensor).contiguous()

            out_tensor: Tensor = self.module(in_tensor)
            out_tensor = torch.clamp_(out_tensor, 0, 1)

            out_tensor = to_hwc_torch(out_tensor)
            out_tensor = flip_r_b_channels_torch(out_tensor)
            out_tensor = out_tensor.float()
            out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return out_img



    @torch.inference_mode()
    def _torch_process_inpaint(
        self,
        in_img: np.ndarray,
        in_mask: np.ndarray
    ) -> np.ndarray:
        in_tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(self.device)
        in_tensor = in_tensor.half() if self.fp16 else in_tensor.float()
        in_tensor = flip_r_b_channels_torch(in_tensor)
        in_tensor = to_nchw_torch(in_tensor).contiguous()

        if len(in_mask.shape) > 2:
            gray = cv2.cvtColor(in_mask, cv2.COLOR_BGR2GRAY)
            _, in_mask = cv2.threshold(gray, 0.5, 1., cv2.THRESH_BINARY)

        in_tensor_mask = torch.from_numpy(np.ascontiguousarray(in_mask))
        in_tensor_mask = in_tensor_mask.to(self.device)
        in_tensor_mask = in_tensor_mask.half() if self.fp16 else in_tensor_mask.float()
        in_tensor_mask = in_tensor_mask / 255.
        in_tensor_mask = to_nchw_torch(in_tensor_mask).contiguous()

        out_tensor: Tensor = self.module(in_tensor, in_tensor_mask)
        out_tensor = torch.clamp_(out_tensor, 0, 1)

        out_tensor = to_hwc_torch(out_tensor)
        out_tensor = flip_r_b_channels_torch(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return out_img



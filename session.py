from __future__ import annotations
import abc
from typing import TypeVar
from warnings import warn

import torch

from .utils.torch_tensor import IdtypeToTorch
from .import_libs import is_cuda_available
from .nn_types import Idtype, NnFrameworkType
from .model import NnModel

if is_cuda_available():
    import cupy as cp

    def set_cupy_cuda_device(device: str = "cuda:0") -> None:
        if not is_cuda_available():
            print(f"[E] No cuda device found, cannot set {device}")
            return
        device_no: int = 0
        try:
            device_no = int(device.split(":")[1])
        except:
            pass
        # print(f"[I] Use cuda device {device_no}")
        cp.cuda.runtime.setDevice(device_no)
else:
    def set_cupy_cuda_device(device: str = "cuda:0") -> None:
        pass


class GenericSession(abc.ABC):
    device: str = 'cpu'

    def __init__(self) -> None:
        super().__init__()
        # self._fp16: bool = False
        self._device: str = 'cpu'
        self.model: NnModel | None = None
        self._dtype: torch.dtype = IdtypeToTorch['fp32']


    def initialize(
        self,
        device: str = 'cpu',
        dtype: Idtype | torch.dtype = 'fp32',
    ):
        if dtype not in self.model.arch.dtypes:
            raise ValueError(
                f"{dtype} is not a valid datatype for the inference session, model arch={self.model.arch_name}"
            )
        self.dtype = dtype
        self.device = device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


    @dtype.setter
    def dtype(self, dtype: Idtype | torch.dtype) -> None:
        self._dtype = (
            dtype if isinstance(dtype, torch.dtype) else IdtypeToTorch[dtype]
        )


    @property
    def device(self) -> str:
        return self._device


    @device.setter
    def device(self, device: str) -> None:
        self._device = device


    @property
    def fwk_type(self) -> NnFrameworkType:
        if self.model is None:
            raise ValueError(f"[E] No valid model for session {self.__class__.__name__}")
        return self.model.fwk_type


    def is_size_valid(
        self,
        size_or_shape: tuple[int, int, int] | tuple[int, int],
        is_shape: bool = True
    ) -> bool:
        return self.model.is_size_valid(size_or_shape, is_shape)


from .nn_onnx.inference.session import OnnxSession
from .nn_pytorch.inference.session import PyTorchSession

# ModelSession = OnnxSession | PyTorchSession
NnModelSession = TypeVar("NnModelSession", OnnxSession, PyTorchSession)

try:
    from .nn_tensor_rt.inference.session import TensorRtSession
    # ModelSession = OnnxSession | PyTorchSession | TensorRtSession
    NnModelSession = TypeVar(
        "NnModelSession",
        OnnxSession,
        PyTorchSession,
        TensorRtSession
    )
except:
    pass




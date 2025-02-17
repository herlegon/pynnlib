from __future__ import annotations
import abc
from typing import TypeVar
from warnings import warn

import torch
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
        self._i_dtype: Idtype = 'fp32'
        self.model: NnModel | None = None


    def initialize(
        self,
        device: str = 'cpu',
        dtype: Idtype = 'fp32',
    ):
        warn("refactor this to remove fp16 flag")
        # if self.model is not None:
        #     self.fp16 = (
        #         dtype == 'fp16'
        #         and (
        #             'fp16' in self.model.dtypes
        #             or self.model.fp16
        #         )
        #     )
        # else:
        #     self.fp16 = False
        if dtype not in self.model.dtypes:
            raise ValueError(
                f"{dtype} is not a valid datatype for the inference session, model arch={self.model.arch_name}"
            )
        self.device = device


    @property
    def i_dtype(self) -> Idtype:
        return self._i_dtype


    @i_dtype.setter
    def i_dtype(self, dtype: Idtype) -> None:
        self._i_dtype = dtype


    # @property
    # def fp16(self) -> bool:
    #     return self._fp16


    # @fp16.setter
    # def fp16(self, enable: bool) -> None:
    #     self._fp16 = enable


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




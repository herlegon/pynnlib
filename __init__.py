from .logger import nnlogger
from .import_libs import *
if not is_cuda_available():
    print("Error: torch and torchvision must be installed and a CUDA device is mandatory.")
    # raise SystemError("Error: torch and torchvision must be installed and a CUDA device is mandatory.")

if not is_tensorrt_available():
    print("Warning: no CUDA device detected, tensorRT is not available")
else:
    print("CUDA device detected, tensorRT is available")

from .core import nn_lib as nnlib

from .model import (
    OnnxModel,
    PytorchModel,
    TrtModel,
    ModelExecutor,
    NnModel,
    TrtEngine,
)

from .nn_onnx.inference.session import OnnxSession
from .nn_pytorch.inference.session import PyTorchSession
from .session import (
    set_cupy_cuda_device,
    NnModelSession
)

try:
    from .nn_tensor_rt.inference.session import TensorRtSession
except:
    TensorRtSession = None

from .nn_tensor_rt.trt_types import ShapeStrategy

from .nn_types import (
    NnFrameworkType,
    NnModelObject,
    Idtype,
)

from .framework import (
    get_supported_model_extensions,
)

from .utils.torch_tensor import (
    torch_dtype_to_np,
    torch_to_cp_dtype,
)

from .utils.tensor import *
from .utils.torch_tensor import (
    flip_r_b_channels,
    IdtypeToTorch,
    np_dtype_to_torch,
    to_nchw,
    to_hwc,
)

__all__ = [
    "nnlogger",
    "nnlib",
    "NnFrameworkType",

    "get_supported_model_extensions",

    "NnModel",
    "OnnxModel",
    "PyTorchModel",
    "TrtModel",

    "NnModelObject",
    "Idtype",
    "IdtypeToTorch",

    "NnModelSession",
    "OnnxSession",
    "PyTorchSession",
    "TensorRtSession",

    "ModelExecutor",
    "ShapeStrategy",

    "is_cuda_available",
    "is_tensorrt_available",
    "set_cupy_cuda_device",

    # For advanced user and dev
    "HostDeviceMemory",
    "torch_to_cp_dtype",
    "torch_dtype_to_np",
    "np_dtype_to_torch",
    "flip_r_b_channels",
    "to_nchw",
    "to_hwc",
    "MemcpyKind",
    "TrtEngine",
]

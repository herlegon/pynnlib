from .logger import nnlogger
from .import_libs import *
if not is_cuda_available():
    raise SystemError("Error: torch and torchvision must be installed")

from .core import nn_lib as nnlib

from .model import (
    OnnxModel,
    PytorchModel,
    TrtModel,
    ModelExecutor,
    NnModel,
)

from .onnx.inference.session import OnnxSession
from .pytorch.inference.session import PyTorchSession
from .session import (
    set_cuda_device,
    NnModelSession
)

try:
    from .tensor_rt.inference.session import TensorRtSession
except:
    TensorRtSession = None

from .tensor_rt.trt_types import ShapeStrategy

from .nn_types import (
    NnFrameworkType,
    NnModelObject,
)

from .framework import (
    get_supported_model_extensions,
)

from .utils.torch_tensor import (
    np_to_torch_dtype
)

from .utils.tensor import *

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

    "NnModelSession",
    "OnnxSession",
    "PyTorchSession",
    "TensorRtSession",

    "ModelExecutor",
    "ShapeStrategy",

    "is_cuda_available",
    "is_tensorrt_available",
    "set_cuda_device",

    # For advanced user and dev
    "HostDeviceMemory",
    "np_to_torch_dtype",
    "flip_r_b_channels",
    "to_nchw",
    "to_hwc",
    "MemcpyKind"

]

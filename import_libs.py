from .logger import nnlogger

__is_tensorrt_available__: bool = False
try:
    nnlogger.debug("[V] Try loading tensorrt package")
    import tensorrt as trt

    __is_tensorrt_available__ = True
    nnlogger.debug(f"[I] Tensorrt package loaded (version {trt.__version__})")

except ModuleNotFoundError:
    nnlogger.debug("[W] Tensorrt package not found")


HAS_PYTORCH_PACKAGE: bool = False
__is_cuda_available__: bool = False
try:
    import torch
    import torchvision
    HAS_PYTORCH_PACKAGE = True
    nnlogger.debug(f"[I] PyTorch package loaded (version {torch.__version__})")
    __is_cuda_available__ = torch.cuda.is_available()

except ModuleNotFoundError:
    nnlogger.debug("[W] Failed to load PyTorch package")



def is_tensorrt_available() -> bool:
    return __is_tensorrt_available__


def is_cuda_available() -> bool:
    return __is_cuda_available__

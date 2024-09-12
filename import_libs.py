from pprint import pprint
from .logger import nnlogger

__is_tensorrt_available__: bool = False
try:
    nnlogger.debug("[V] Try loading tensorrt package")
    import tensorrt as trt
    import sys
    # pprint(sys.modules.keys())
    modules = set(sys.modules) & set(globals())
    module_names = [sys.modules[m] for m in modules]
    if 'tensorrt' in module_names:
        print(f"in modules: [{module_names}]")
        __is_tensorrt_available__ = True

    # if 'trt' in sys.modules or 'tensorrt' in sys.modules:
    #     # print(trt.__version__)
    #     __is_tensorrt_available__ = True
    # nnlogger.debug(f"[I] Tensorrt package loaded (version {trt.__version__})")

except ModuleNotFoundError:
    nnlogger.debug("[W] Tensorrt package not found")

try:
    print(trt.__version__)
    __is_tensorrt_available__ = True
except:
    __is_tensorrt_available__ = False


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

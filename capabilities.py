# TODO: move this to 'system' submodule
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
import sys
from .import_libs import (
    is_tensorrt_available,
    is_cuda_available,
)
from .framework import NnFrameworkType


class ExecutionProvider(Enum):
    CPU = 'cpu'
    CUDA = 'CUDA'
    TENSORRT = 'TensorRT'



@dataclass
class FwkCapabilities:
    parse: bool = False
    inference: bool = False


_is_platform_supported: bool = (
    sys.platform == "win32"
    or sys.platform == "linux"
)

try:
    from system import GpuDevices

    # PyTorch: https://pytorch.org/
    # https://github.com/microsoft/DirectML#hardware-requirements
    # https://www.amd.com/en/technologies/vulkan

    # DirectML:
    # https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#requirements
    # DirectML was introduced in Windows 10, version 1903
    # NVIDIA Kepler (GTX 600 series) and above
    # AMD GCN 1st Gen (Radeon HD 7000 series) and above
    # Intel Haswell (4th-gen core) HD Integrated Graphics and above


    def get_system_capabilities() -> dict[NnFrameworkType, dict[ExecutionProvider, bool]]:

        capabilities: dict[NnFrameworkType, dict[ExecutionProvider, FwkCapabilities]] = {}
        gpu_devices = GpuDevices()

        if is_cuda_available():
            # PyTorch: https://pytorch.org/
            capabilities[NnFrameworkType.PYTORCH] = {
                ExecutionProvider.CPU: _is_platform_supported,

                ExecutionProvider.CUDA: (
                    _is_platform_supported
                    and is_cuda_available()
                    and gpu_devices.supports_fp_dtype('fp16', 'nvidia')
                ),

                ExecutionProvider.TENSORRT: (
                    _is_platform_supported
                    and is_tensorrt_available()
                    and gpu_devices.supports_fp_dtype('fp16', 'nvidia')
                ),

            }


        # Use ONNX runtime / tensorrt /directML
        capabilities[NnFrameworkType.ONNX] = {
            ExecutionProvider.CPU: _is_platform_supported,

            ExecutionProvider.CUDA: (
                # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
                _is_platform_supported
                and is_cuda_available()
                and gpu_devices.supports_fp_dtype('fp16', 'nvidia')
            ),

            ExecutionProvider.TENSORRT: (
                _is_platform_supported
                and is_tensorrt_available()
                and gpu_devices.supports_fp_dtype('fp16', 'nvidia')
            ),

        }

        if is_tensorrt_available():
            capabilities[NnFrameworkType.TENSORRT] = {
                ExecutionProvider.TENSORRT: (
                    _is_platform_supported
                    and is_tensorrt_available()
                    and gpu_devices.supports_fp_dtype('fp16', 'nvidia')
                ),
            }
        return capabilities

except:
    # Default when used outside of Herlegon system. No support
    HAS_CUDA_DEVICE: bool = False
    if is_cuda_available():
        import torch
        HAS_CUDA_DEVICE = torch.cuda.is_available()

    def get_system_capabilities() -> dict[NnFrameworkType, dict[ExecutionProvider, bool]]:
        capabilities: dict[NnFrameworkType, dict[ExecutionProvider, FwkCapabilities]] = {
            NnFrameworkType.PYTORCH: {
                ExecutionProvider.CPU: _is_platform_supported,
                ExecutionProvider.CUDA: _is_platform_supported and is_cuda_available() and HAS_CUDA_DEVICE,
                ExecutionProvider.TENSORRT: _is_platform_supported and is_tensorrt_available() and HAS_CUDA_DEVICE,
            },
            NnFrameworkType.ONNX: {
                ExecutionProvider.CPU: _is_platform_supported,
                ExecutionProvider.CUDA: _is_platform_supported and is_cuda_available() and HAS_CUDA_DEVICE,
                ExecutionProvider.TENSORRT: _is_platform_supported and is_tensorrt_available() and HAS_CUDA_DEVICE,
            },
            NnFrameworkType.TENSORRT: {
                ExecutionProvider.TENSORRT: _is_platform_supported and is_tensorrt_available() and HAS_CUDA_DEVICE,
            },
        }
        return capabilities


def print_capabilities():
    print("Capabilities\n--------------------------------------")
    capabilities = get_system_capabilities()
    for fwk, runtime in capabilities.items():
        print(f"""{fwk.value}: {', '.join(
            [xp.value for xp, capability in runtime.items() if capability.inference])}""")

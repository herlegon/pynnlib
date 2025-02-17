from argparse import ArgumentParser, RawTextHelpFormatter
import os
import re
import signal
import sys
import time
from typing import Any

if not os.path.exists("pynnlib"):
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if os.path.exists(os.path.join(root_path, "pynnlib")):
        sys.path.append(root_path)

from pynnlib import (
    Idtype,
    nnlib,
    NnModel,
    TrtModel,
    ShapeStrategy,
    is_cuda_available,
    is_tensorrt_available,
)
from pynnlib.utils import absolute_path, path_split
from pynnlib.utils.p_print import *



def convert_to_tensorrt(
    arguments: Any,
    model: NnModel,
    device: str,
    dtype: Idtype,
) -> TrtModel | None:
    trt_model: TrtModel | None = None

    # Shape strategy
    shape_strategy: ShapeStrategy = ShapeStrategy()
    def _str_to_size(size_str: str) -> tuple[int, int] | None:
        if (match := re.match(re.compile(r"^(\d+)x(\d+)$"), size_str)):
            return (int(match.group(1)), int(match.group(2)))
        return None

    opt_size = _str_to_size(arguments.opt_size)
    if opt_size is None:
        sys.exit(red(f"[E] Erroneous option: {arguments.opt_size}"))
    shape_strategy.opt_size = opt_size

    if not arguments.fixed_size:
        min_size = _str_to_size(arguments.min_size)
        if min_size is None:
            sys.exit(red(f"[E] Erroneous option: {arguments.min_size}"))
        shape_strategy.min_size = min_size

        max_size = _str_to_size(arguments.max_size)
        if max_size is None:
            sys.exit(red(f"[E] Erroneous option: {arguments.max_size}"))
        shape_strategy.max_size = max_size

    else:
        shape_strategy.min_size = shape_strategy.opt_size
        shape_strategy.max_size = shape_strategy.opt_size

    if not shape_strategy.is_valid():
        sys.exit(red(f"[E] Erroneous sizes"))

    # Optimization
    opt_level = arguments.opt_level
    opt_level = None if not 1 <= opt_level <= 5 else opt_level

    trt_model: TrtModel = nnlib.convert_to_tensorrt(
        model=model,
        shape_strategy=shape_strategy,
        dtype=dtype,
        optimization_level=opt_level,
        opset=arguments.opset,
        device=device,
        out_dir=path_split(model.filepath)[0],
    )

    return trt_model



def main():
    parser = ArgumentParser(
        description="Convert model into Onnx model or TensorRT engine",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='',
        required=True,
        help="""Model (PyTorch, ONNX) to convert.
\n"""
    )

    parser.add_argument(
        "-onnx",
        "--onnx",
        action="store_true",
        required=False,
        default=False,
        help="""Save model as an Onnx model.
\n"""
    )

    parser.add_argument(
        "-trt",
        "--trt",
        action="store_true",
        required=False,
        default=False,
        help="""Generates a TensorRT engine.
\n"""
    )

    parser.add_argument(
        "-fp32",
        "--fp32",
        action="store_true",
        required=False,
        default=True,
        help="""Support full precision (fp32).
\n"""
    )

    parser.add_argument(
        "-fp16",
        "--fp16",
        action="store_true",
        required=False,
        default=False,
        help="""Support half precision (fp16).
\n"""
    )

    parser.add_argument(
        "-bf16",
        "--bf16",
        action="store_true",
        required=False,
        default=False,
        help="""Support mixed precision (bf16).
\n"""
    )

    parser.add_argument(
        "-opset",
        "--opset",
        type=int,
        required=False,
        default=20,
        help="""Onnx opset version. Used when converting a PyTorch Model to onnx/tensorrt.
\n"""
    )
    parser.add_argument(
        "--static",
        "-static",
        action="store_true",
        required=False,
        default=False,
        help="""(TensorRT) Use a static shape (opt_size) when converting the model.
\n"""
    )
    parser.add_argument(
        "-min",
        "--min_size",
        type=str,
        default='64x64',
        required=False,
        help="""(TensorRT) min. size used to generate a tensorRT engine.
format: WxH
\n"""
    )
    parser.add_argument(
        "-opt",
        "--opt_size",
        type=str,
        default='768x576',
        required=False,
        help="""(TensorRT) opt. size used to generate a tensorRT engine.
format: WxH.
use the input video dimension if set to \'input\'.
\n"""
    )
    parser.add_argument(
        "-max",
        "--max_size",
        type=str,
        default='1920x1080',
        required=False,
        help="""(TensorRT) max. size used to generate a tensorRT engine.
format: WxH.
\n"""
    )
    parser.add_argument(
        "-fixed",
        "--fixed_size",
        action="store_true",
        required=False,
        default=False,
        help="""(TensorRT) use the opt_size for both min_size and max_size.
\n"""
    )
    parser.add_argument(
        "--opt_level",
        type=int,
        default=3,
        required=False,
        help="""(TensorRT) (not yet supported) Optimization level. [1..5].
\n"""
    )

    arguments = parser.parse_args()

    if not arguments.trt and not arguments.onnx:
        sys.exit(red(f"[E] at least --onnx or --trt must be specified"))

    if arguments.fp16 and not is_cuda_available():
        sys.exit(red(f"[E] No CUDA device found, cannot convert with fp16 support"))

    if arguments.bf16 and not is_cuda_available():
        sys.exit(red(f"[E] No CUDA device found, cannot convert with bf16 support"))

    # device and datatype
    device = "cuda" if is_cuda_available() else 'cpu'

    # Open model file
    model_filepath: str = absolute_path(arguments.model)
    if not os.path.isfile(model_filepath):
        sys.exit(red(f"[E] {model_filepath} is not a valid file"))
    model: NnModel = nnlib.open(model_filepath, device)
    print(lightgreen(f"[I] arch: {model.arch_name}"))
    print(model)

    if arguments.fp16 and not 'fp16' in model.arch.dtypes:
        sys.exit(red(f"[W] This arch does not support conversion with fp16 support"))
    fp16: bool = arguments.fp16 and 'fp16' in model.arch.dtypes

    if arguments.bf16 and not 'bf16' in model.arch.dtypes:
        sys.exit(red(f"[W] This arch does not support conversion with bf16 support"))
    bf16: bool = arguments.bf16 and 'bf16' in model.arch.dtypes

    # bf16 has the priority if multiple types are provided
    c_dtype: Idtype = 'fp32'
    if fp16:
        c_dtype = 'fp16'
    if bf16:
        c_dtype = 'bf16'

    # Model conversion
    model = model
    if arguments.onnx:
        print(f"[V] Convert {model.filepath} to ONNX (dtype={c_dtype}): ")
        start_time = time.time()
        onnx_model = nnlib.convert_to_onnx(
            model=model,
            opset=arguments.opset,
            dtype=c_dtype,
            static=arguments.static,
            device=device,
            out_dir=path_split(model.filepath)[0],
        )
        elapsed_time = time.time() - start_time
        print(lightgreen(f"[I] Onnx model saved as {onnx_model.filepath}"))
        print(f"[V] Converted in {elapsed_time:.2f}s")
        print(onnx_model)

    if arguments.trt:
        if is_tensorrt_available():
            print(f"[V] Convert {model.filepath} to TensorRT (c_dtype={c_dtype}): ")
            start_time = time.time()
            trt_model = convert_to_tensorrt(
                arguments, model=model, device=device, dtype=c_dtype,
            )
            if trt_model is None:
                print(red("[E] Failed to convert to a TensorRT engine"))
            else:
                elapsed_time = time.time() - start_time
                print(lightgreen(f"[I] TensorRT engine saved as {trt_model.filepath}"))
                print(f"[V] Converted in {elapsed_time:.2f}s")
        else:
            print(red("[E] No compatible device found, cannot convert to an TensorRT engine"))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

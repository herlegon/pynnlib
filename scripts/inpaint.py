from argparse import ArgumentParser, RawTextHelpFormatter
import logging
import logging.config
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any
import cv2
import numpy as np

from pynnlib.nn_types import Idtype

# logging.config.fileConfig('config.ini')
# logger = logging.getLogger("pynnlib")
# logging.basicConfig(filename="logs.log", filemode="w", format="%(name)s → %(levelname)s: %(message)s")
# logging.config.fileConfig('config.ini')

if not os.path.exists("pynnlib"):
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if os.path.exists(os.path.join(root_path, "pynnlib")):
        sys.path.append(root_path)

from pynnlib import (
    nnlogger,
    nnlib,
    NnModel,
    is_cuda_available,
    NnModelSession,
    NnFrameworkType,
    ShapeStrategy,
    TrtModel
)
from pynnlib.utils import absolute_path, path_split
from pynnlib.utils.p_print import *



def convert_to_tensorrt(
    arguments: Any,
    model: NnModel,
    device: str,
    dtype: Idtype = 'fp32',
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
        tf32=False,
        optimization_level=opt_level,
        opset=arguments.opset,
        device=device,
        out_dir=path_split(model.filepath)[0],
    )

    return trt_model



def load_image_fp32(filepath: Path | str) -> np.ndarray:
    # Simplified function for testing purpose only
    try:
        img: np.ndarray = cv2.imdecode(
            np.fromfile(filepath, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED
        )
    except Exception as e:
        raise RuntimeError(red(type(e)))
    div: float = float(np.iinfo(img.dtype).max)
    img = img.astype(np.float32)
    img /= div
    return img


def load_image(filepath: Path | str) -> np.ndarray:
    # Simplified function for testing purpose only
    try:
        img: np.ndarray = cv2.imdecode(
            np.fromfile(filepath, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED
        )
    except Exception as e:
        raise RuntimeError(red(type(e)))
    return img


def write_image(filepath: Path | str, img: np.ndarray) -> None:
    # Simplified function for testing purpose only: save as 8-bit image
    # Overwrite input image
    extension = os.path.splitext(filepath)[1]
    if img.dtype == np.float32:
        img = img.clip(0., 1.)
        img *= 255.
        img = img.astype(np.uint8)
    if img.dtype != np.uint8:
        raise ValueError(f"Unsupported input image dtype: {img.dtype}")
    try:
        _, img_buffer = cv2.imencode(f".{extension}", img)
        with open(filepath, "wb") as buffered_writer:
            buffered_writer.write(img_buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to save image as {filepath}. {type(e)}")



def main():
    parser = ArgumentParser(
        description="Perform an inference on an image.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--img",
        type=str,
        default='',
        required=True,
        help="Input image."
    )

    parser.add_argument(
        "--mask",
        type=str,
        default='',
        required=True,
        help="Mask."
    )

    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default='_out',
        required=False,
        help="Append this text to the output image filename."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='',
        required=True,
        help="Model: PyTorch, ONNX, TensorRT."
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        required=False,
        default=True,
        help="""Use the first cuda device as the execution provider.
Fallback to cpu if no CUDA device found.
\n"""
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        required=False,
        default=False,
        help="""Use the CPU as the execution provider.
Ignored if the model is a TensorRT engine."
\n"""
    )

    parser.add_argument(
        "--dml",
        action="store_true",
        required=False,
        default=False,
        help="""Use the DirectML as the execution provider.
Ignored if the model is not an onnx model."
\n"""
    )

    parser.add_argument(
        "-fp32",
        "--fp32",
        action="store_true",
        required=False,
        default=True,
        help="""Full precision (fp32)
\n"""
    )

    parser.add_argument(
        "-fp16",
        "--fp16",
        action="store_true",
        required=False,
        default=False,
        help="""Half precision (fp16).
Fallback to float if the execution provider does not support it
\n"""
    )

    parser.add_argument(
        "-n",
        "--n",
        type=int,
        required=False,
        default=1,
        help="Repeat \'n\' times the inference with the same image"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Print the model info"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Used to debug"
    )

    parser.add_argument(
        "--profiling",
        action="store_true",
        required=False,
        help="for profiling"
    )


    arguments = parser.parse_args()

    if arguments.debug:
        # FileOutputHandler = logging.FileHandler('logs.log', mode='w')
        # nnlogger.addHandler(FileOutputHandler)
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")

    # Open image/mask as float, the inference session converts
    # the datatype as needed
    in_img: np.ndarray | None = None
    in_img_fp = absolute_path(arguments.img)
    try:
        in_img = load_image_fp32(in_img_fp)
    except:
        sys.exit(red(f"Failed to open image: {arguments.img}"))

    in_mask: np.ndarray | None = None
    in_mask_fp = absolute_path(arguments.mask)
    try:
        # in_mask = load_image_fp32(in_mask_fp)
        in_mask = load_image(in_mask_fp)
    except:
        sys.exit(red(f"Failed to open mask: {arguments.mask}"))

    # Output image filepath:
    dir, basename, ext = path_split(in_img_fp)
    out_img_fp: str = os.path.join(dir, f"{basename}{arguments.suffix}{ext}")

    print(lightcyan(f"Input image:"), f"{in_img_fp}")
    print(lightcyan(f"Mask:"), f"{in_mask_fp}")
    print(lightcyan(f"Output image:"), f"{out_img_fp}")

    # Select a device to run the inference
    # For a tensorrt engine, the device has to be a cuda device
    device_for_parse: str = (
        "cuda"
        if is_cuda_available() and arguments.cuda
        else 'cpu'
    )
    print(lightcyan(f"Use device for detection:"), f"{device_for_parse}")
    # Open model file
    model_filepath: str = absolute_path(arguments.model)
    if not os.path.isfile(model_filepath):
        sys.exit(red(f"{model_filepath} does not exist."))
    start_time= time.time()
    try:
        model: NnModel = nnlib.open(model_filepath, device_for_parse)
    except:
        model: NnModel = nnlib.open(model_filepath, device_for_parse)
        sys.exit(red(f"Failed to open model: {arguments.model}"))
    elapsed = time.time() - start_time
    print(lightcyan(f"Model:"), f"{model.filepath}")
    print(
        lightcyan(f"\tarch:"), model.arch_name,
        lightcyan(f"scale:"), model.scale,
        lightcyan(f"datatypes:"), ', '.join(model.dtypes),
        f"\t\t(parsed in {1000 * elapsed:.1f}ms)"
    )
    if arguments.verbose:
        print(model)


    # Verify that the image is valid:
    if not model.is_size_valid(in_img.shape):
        h, w = in_img.shape[:2]
        sys.exit(red(f"Image size ({w}x{h}) is not supported by the model."))

    # Execution provider for inference: set device
    if (
        (not arguments.cpu and not arguments.dml and not arguments.cuda)
        and model.fwk_type != NnFrameworkType.TENSORRT
    ):
        sys.exit(red(f"An execution provider must be specified."))

    device: str = 'cpu'
    if arguments.dml:
        device = 'dml'
    elif arguments.cuda or model.fwk_type == NnFrameworkType.TENSORRT:
        device = f"cuda:0"

    # Check datatype supported by the execution provider and by the model
    fp16: bool = arguments.fp16
    if device == 'cpu' and fp16:
        sys.exit(red("The execution provider does not support Half datatype (fp16)."))
    if fp16 and 'fp16' not in model.arch.dtypes:
        sys.exit(red("This model does not support Half datatype (fp16)."))

    dtype: Idtype = 'fp32'
    if fp16:
        dtype = 'fp16'

    # Create a session
    if arguments.verbose:
        print("Create a session")
    session: NnModelSession = nnlib.session(model)
    if arguments.verbose:
        print("Initialize the session")
    try:
        session.initialize(device=device, dtype=dtype)
    except Exception as e:
        session.initialize(device=device, dtype=dtype)
        sys.exit(red(f"Error: {e}"))

    print(lightcyan(f"Inference with"), f"{model.filepath}")
    print(lightcyan(f"\tScale:"), f"{model.scale}")
    print(lightcyan(f"\tFramework:"), f"{model.fwk_type.value}")
    print(lightcyan(f"\tDevice:"), f"{device}")
    print(lightcyan(f"\tDatatype:"), f"{dtype}")

    # Inference
    inferences: int = arguments.n
    if inferences > 1:
        print(lightcyan(f"Repeat inference"), inferences, lightcyan("times"))

    if arguments.profiling:
        print("start", flush=True)
        time.sleep(3)


    start_time= time.time()
    for _ in range(inferences):
        # in_img = in_img[:1024,:1024,:]
        # in_mask = in_mask[:1024,:1024]
        out_img: np.ndarray = session.process(in_img, in_mask)
    elapsed = time.time() - start_time

    if inferences > 1:
        print(lightcyan(f"{inferences} inferences:"), yellow(f"{1000 * elapsed:.1f}ms"))
        print(lightcyan(f"fps:"), yellow(f"{float(inferences)/elapsed:.1f}fps"))
    else:
        print(lightcyan(f"Inference:"), yellow(f"{1000 * elapsed:.1f}ms"))

    if arguments.profiling:
        return
    # Save image
    try:
        write_image(out_img_fp, out_img)
    except:
        sys.exit(red(f"Failed to save image as {out_img_fp}"))
    print(lightgreen(f"Image saved as {out_img_fp}"))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

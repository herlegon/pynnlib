from __future__ import annotations
import onnx
from pynnlib.import_libs import is_tensorrt_available
from pynnlib.model import OnnxModel
if is_tensorrt_available():
    from pynnlib.nn_tensor_rt.trt_types import ShapeStrategy, TensorrtModel
    from pynnlib.nn_tensor_rt.model.onnx_to_trt import onnx_to_trt_engine
else:
    # print("[W] TensorRT is not supported: model cannot be converted")
    def onnx_to_trt_engine(*args):
        raise RuntimeError("TensorRT is not supported")


def to_tensorrt(
    model: OnnxModel,
    device: str,
    fp16: bool,
    bf16: bool,
    shape_strategy: ShapeStrategy,
) -> TensorrtModel:

    return onnx_to_trt_engine(
        model,
        device,
        fp16,
        bf16,
        shape_strategy
    )


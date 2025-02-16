from __future__ import annotations
from pynnlib.import_libs import is_tensorrt_available
from pynnlib.model import OnnxModel
from pynnlib.nn_types import Idtype
if is_tensorrt_available():
    from pynnlib.nn_tensor_rt.trt_types import ShapeStrategy, TensorrtModel
    from pynnlib.nn_tensor_rt.archs.onnx_to_trt import onnx_to_trt_engine
else:
    # print("[W] TensorRT is not supported: model cannot be converted")
    def onnx_to_trt_engine(*args):
        raise RuntimeError("TensorRT is not supported")


def to_tensorrt(
    model: OnnxModel,
    device: str,
    dtypes: set[Idtype],
    shape_strategy: ShapeStrategy,
) -> TensorrtModel:

    return onnx_to_trt_engine(
        model=model,
        device=device,
        dtypes=dtypes,
        shape_strategy=shape_strategy
    )


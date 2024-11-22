from __future__ import annotations
from io import BytesIO
from pprint import pprint
import onnx
import torch
from typing import TYPE_CHECKING

from pynnlib.architecture import SizeConstraint
from pynnlib.import_libs import is_cuda_available
from ..inference.session import PyTorchSession
if TYPE_CHECKING:
    from pynnlib.model import PytorchModel
from pynnlib.utils.p_print import *


def to_onnx(
    model: PytorchModel,
    fp16: bool,
    opset: int,
    static: bool = False,
    device: str = 'cpu',
    batch: int = 1,
) -> onnx.ModelProto | None:
    """Returns an Onnx model as a byte buffer
        if size is not None, use it to convert to a static shape
    """
    print(f"[V] PyTorch to ONNX")

    try:
        session: PyTorchSession = model.arch.create_session(model)
    except:
        raise NotImplementedError(red(f"{model.arch.name} is not supported"))
        return None

    if not is_cuda_available():
        print("[W] cuda not available, fallback to cpu")
        device = 'cpu'
        fp16 = False

    # https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-convert-model
    print(f"[V] PyTorch to ONNX: initialize a session, device={device}, fp16={fp16}, opset={opset}")
    session.initialize(device=device, fp16=fp16)

    # https://github.com/onnx/onnx/issues/654
    if batch == 1:
        dynamic_axes = {
            'input': {2: "height", 3: "width"},
            'output': {2: "height", 3: "width"},
        }
    else:
        dynamic_axes = {
            'input': {0: "batch", 2: "height", 3: "width"},
            'output': {0: "batch", 2: "height", 3: "width"},
        }

    size: SizeConstraint | None = model.size_constraint
    w, h = size.min if size is not None and size.min is not None else (32, 32)

    dummy_input = torch.rand(
        batch,
        model.in_nc,
        h,
        w,
        device=device,
        requires_grad=True
    )
    dummy_input = dummy_input.half() if fp16 else dummy_input.float()

    model_proto: onnx.ModelProto
    with BytesIO() as bytes_io:
        torch.onnx.export(
            session.module,
            dummy_input,
            bytes_io,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes if not static else None,
        )
        bytes_io.seek(0)
        model_proto = onnx.load(bytes_io)

    # Convert float to float16 ?
    # Not really intersting
    # from onnxconverter_common import float16
    # if fp16 and device != 'cpu':
    #     model_proto = float16.convert_float_to_float16(model_proto)

    try:
        onnx.checker.check_model(model_proto)
    except Exception as e:
        print(f"[E] Converted Onnx model is not valid: {type(e)}")
        return None

    print(f"[V] ONNX model proto generated")

    return model_proto


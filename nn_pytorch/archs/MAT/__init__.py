from __future__ import annotations
from pprint import pprint
from pynnlib.nn_pytorch.archs import contains_any_keys
from pynnlib.utils.p_print import *
from pynnlib.architecture import (
    InferType,
    NnPytorchArchitecture,
)
from pynnlib import is_cuda_available
from pynnlib.model import PytorchModel, SizeConstraint
from ...inference.session import PyTorchSession
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from .module.mat import MAT
from .module.bias_act import compile_bias_act_ext
from .module.upfirdn2d import compile_upfirdn2d_ext
from io import BytesIO
import onnx
import torch


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 1
    in_nc: int = 3
    out_nc: int = 3

    verbose: bool = False
    compile_bias_act_ext(verbose=verbose)
    compile_upfirdn2d_ext(verbose=verbose)

    for k in list(state_dict.keys()).copy():
        if k.startswith(("synthesis.", "mapping.")):
            state_dict[f"model.{k}"] = state_dict.pop(k)

    # Update model parameters
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=MAT,
    )



def to_onnx_inpaint(
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
            "mask":  {2: "height", 3: "width"},
            'output': {2: "height", 3: "width"},
        }
    else:
        dynamic_axes = {
            'input': {0: "batch", 2: "height", 3: "width"},
            "mask":  {0: "batch", 2: "height", 3: "width"},
            'output': {0: "batch", 2: "height", 3: "width"},
        }

    size: SizeConstraint | None = model.size_constraint
    w, h = size.min if size is not None and size.min is not None else (32, 32)

    w, h = 1440, 1080

    dummy_input = torch.rand(
        batch,
        model.in_nc,
        h,
        w,
        device=device,
        requires_grad=True
    )
    dummy_input = dummy_input.half() if fp16 else dummy_input.float()

    dummy_mask = torch.rand(
        batch,
        1,
        h,
        w,
        device=device,
        requires_grad=True
    )
    dummy_mask = dummy_mask.half() if fp16 else dummy_mask.float()

    static = True

    model_proto: onnx.ModelProto
    with BytesIO() as bytes_io:
        torch.onnx.export(
            session.module,
            (dummy_input, dummy_mask),
            bytes_io,
            export_params=True,
            input_names=['input', 'mask'],
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


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='MAT inpainting',
        detection_keys=(
            "synthesis.first_stage.conv_first.conv.resample_filter",
            "model.synthesis.first_stage.conv_first.conv.resample_filter",
        ),
        detect=contains_any_keys,
        parse=parse,
        # to_onnx=None,
        # to_onnx=to_onnx_inpaint,
        dtypes=set(['fp32', 'fp16']),
        infer_type=InferType(
            type='inpaint',
            inputs=2,
            outputs=1
        )
    ),
)

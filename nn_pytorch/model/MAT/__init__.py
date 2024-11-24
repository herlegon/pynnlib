import math
from pprint import pprint
from typing import Any, Literal
from pynnlib.nn_pytorch.model import contains_any_keys
from pynnlib.utils.p_print import *
from pynnlib.architecture import (
    InferType,
    NnPytorchArchitecture,
)
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx
from .module.mat import MAT


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 1
    in_nc: int = 3
    out_nc: int = 3

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


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='MAT',
        detection_keys=(
            "synthesis.first_stage.conv_first.conv.resample_filter",
            "model.synthesis.first_stage.conv_first.conv.resample_filter",
        ),
        detect=contains_any_keys,
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16']),
        infer_type=InferType(
            inputs=2,
            outputs=1
        )
    ),
)

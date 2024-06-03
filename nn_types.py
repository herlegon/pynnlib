from __future__ import annotations
from enum import Enum
import os
import onnx
from pathlib import PurePath
from typing import Literal, TypeAlias

from .pytorch.torch_types import StateDict
from .tensor_rt.trt_types import TrtEngine


NnArchitectureType: TypeAlias = str

class NnFrameworkType(Enum):
    ONNX = 'ONNX'
    PYTORCH = 'PyTorch'
    TENSORRT = 'TensorRT'


def filepath_to_fwk(filepath: str) -> NnFrameworkType:
    # Cannot use _value2member_map_ because of lower/upper case
    value = PurePath(os.path.dirname(filepath)).parts[-1]
    enum_dict = {t.value.lower(): t.value for t in NnFrameworkType}
    try:
        return NnFrameworkType._value2member_map_[enum_dict[value]]
    except:
        raise ValueError(f"[E] {value} is not a valid framework key")




NnModelObject = onnx.ModelProto | StateDict | TrtEngine

# Supported datatypes for model
NnModelDtype = Literal['fp32', 'fp16', 'bf16']

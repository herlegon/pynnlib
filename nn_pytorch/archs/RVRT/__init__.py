import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import (
    get_max_indice,
    get_pixelshuffle_params
)
from .module.network_rvrt import RVRT
from ..torch_to_onnx import to_onnx
from ...torch_types import StateDict



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict




MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='RVRT',
        detection_keys=(
            "tututuututut",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

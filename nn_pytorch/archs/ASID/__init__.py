import math
import re
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from .module.ASID import ASID


def _get_max_indice(state_dict: StateDict, key: str) -> int:
    indice_set: set = set()
    for k in state_dict:
        if r := re.search(re.compile(rf"{key}(\d+).res_end.weight"), k):
            indice_set.add(int(r.group(1)))
    return max(indice_set)


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    out_nc: int = 3
    in_nc: int = out_nc

    scale: int = math.isqrt(state_dict["up.0.weight"].shape[0] // 3)
    res_num = _get_max_indice(state_dict, "block") + 1
    is_d8: bool = bool(res_num == 8)
    num_feat: int = state_dict["block0.res_end.bias"].shape[0]

    # Detectable?
    block_num: int = 1
    bias: bool = True
    pe: bool = True
    window_size: int = 8

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=ASID,
        window_size=window_size,
        num_feat=num_feat,
        res_num=res_num,
        block_num=block_num,
        bias=bias,
        pe=pe,
        d8=is_d8,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="ASIDd8",
        detection_keys=(
            "block7.res_end.weight"
            "block2.res_end.weight",
            "block0.res_end.weight",
            "up.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(32, 32)
        )
    ),
    NnPytorchArchitecture(
        name="ASID",
        detection_keys=(
            "block2.res_end.weight",
            "block0.res_end.weight",
            "up.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(32, 32)
        )
    ),
)

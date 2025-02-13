from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_nsequences,
)
from ..torch_to_onnx import to_onnx
from .module.TMP_arch import TMP


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict

    # default
    num_feat: int = 64
    num_extract_block: int = 3
    hr_in: bool = False

    in_nc, num_feat = state_dict["conv_first.weight"].shape[:2]
    out_nc = in_nc
    # scale is fixed
    scale: int = 4
    num_extract_block = get_nsequences(state_dict, "feature_extraction")

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=TMP,
        num_in_ch=in_nc,
        num_feat=num_feat,
        num_extract_block=num_extract_block,
        hr_in=hr_in
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="TMP",
        detection_keys=(
            "align.convs.0.weight",
            "align.reconstruction.0.conv1.weight",
            "feature_extraction.0.conv1.weight",
            "conv_first.weight",
            "upconv1.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

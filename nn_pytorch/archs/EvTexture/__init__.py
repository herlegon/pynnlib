import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_nsequences,
)
from ..torch_to_onnx import to_onnx
from .module.evtexture import EvTexture


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict

    in_nc: int = state_dict["cnet.main.0.weight"].shape[1]
    out_nc, num_feat = state_dict["conv_last.weight"].shape[:2]
    num_block = get_nsequences(state_dict, "backward_trunk.main.2")
    # scale is fixed
    scale: int = 4

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=EvTexture,
        num_feat=num_feat,
        num_block=num_block,
        spynet_path=""
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="EvTexture",
        detection_keys=(
            "cnet.main.0.weight",
            "enet.conv1.weight",
            "update_block.gru.convz.weight",
            "backward_trunk.main.0.weight",
            "spynet.basic_module.0.basic_module.0.weight",
            "conv_last.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

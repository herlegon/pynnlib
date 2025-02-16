import math
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx
from .module.RTMoSR import RTMoSR


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    unshuffle = False
    if "to_feat.1.alpha" in state_dict:
        unshuffle = True
        scale = math.isqrt(state_dict["to_feat.1.conv_3x3_rep.weight"].shape[1] // 3)
        dim = state_dict["to_feat.1.conv_3x3_rep.weight"].shape[0]
    else:
        scale = math.isqrt(state_dict["to_img.0.conv_3x3_rep.weight"].shape[0] // 3)
        dim = state_dict["to_feat.conv_3x3_rep.weight"].shape[0]

    dccm = "body.0.fc2.alpha" in state_dict
    se = "body.0.conv.2.squeezing.0.weight" in state_dict
    ffn = state_dict["body.0.fc1.conv_3x3_rep.weight"].shape[0] / dim / 2
    n_blocks = get_nsequences(state_dict, "body")

    # Update model parameters
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RTMoSR,
        dim=dim,
        ffn_expansion=ffn,
        n_blocks=n_blocks,
        unshuffle_mod=unshuffle,
        dccm=dccm, se=se
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RTMoSR",
        detection_keys=(
            "body.0.conv.1.conv5x5_reparam.weight",
            "body.0.fc1.conv_3x3_rep.weight",
            "to_img.0.conv_3x3_rep.weight",
            "body.0.fc2.alpha",
            "to_img.0.conv3.eval_conv.weight",
            "body.0.norm.scale",
            "body.0.norm.offset",
            "body.0.fc1.alpha",
            "to_img.0.alpha",
            "to_img.0.conv1.k0",
            "to_img.0.conv1.b0",
            "to_img.0.conv1.k1",
            "to_img.0.conv1.b1",
            "to_img.0.conv2.weight",
            "to_img.0.conv3.sk.weight",
            "to_img.0.conv3.conv.0.weight",
            "to_img.0.conv3.conv.1.weight",
            "to_img.0.conv3.conv.2.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16']),
    ),
)

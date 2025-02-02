import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
)
from .module.OmniSR import OmniSR
from ..torch_to_onnx import to_onnx
from ...torch_types import StateDict


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict

    state_dict_keys = set(state_dict.keys())
    for key in state_dict_keys:
        if key.endswith(("total_ops", "total_params")):
            del state_dict[key]

    # default
    # in_nc: int = 3
    # out_nc: int = 3
    # num_feat: int = 64
    # block_num = 1
    # pe = True
    # window_size = 8
    # res_num = 1
    # scale: int = 4
    # bias = True

    num_feat: int = state_dict["input.weight"].shape[0]
    in_nc: int = state_dict["input.weight"].shape[1]
    bias = "input.bias" in state_dict

    pixelshuffle_shape = state_dict["up.0.weight"].shape[0]
    scale, out_nc = get_scale_and_out_nc(pixelshuffle_shape, in_nc)

    res_num = get_max_indice(state_dict, "residual_layer") + 1
    block_num = get_max_indice(state_dict, "residual_layer.0.residual_layer")

    rel_pos_bias_key = "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
    if rel_pos_bias_key in state_dict:
        pe = True
        # rel_pos_bias_weight = (2 * window_size - 1) ** 2
        rel_pos_bias_weight = state_dict[rel_pos_bias_key].shape[0]
        window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
    else:
        pe = False

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=OmniSR,
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        num_feat=num_feat,
        block_num=block_num,
        pe=pe,
        window_size=window_size,
        res_num=res_num,
        bias=bias,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='OmniSR',
        detection_keys=(
            "residual_layer.0.residual_layer.0.layer.0.fn.0.weight",
            "input.weight",
            "up.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(16, 16)
        )
    ),
)

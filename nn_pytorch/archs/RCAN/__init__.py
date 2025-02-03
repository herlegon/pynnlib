import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
)
from .module.rcan import RCAN
from ..torch_to_onnx import to_onnx
from ...torch_types import StateDict


def get_pixelshuffle_params(
    state_dict,
    upsample_key: str = "upsample",
    default_nf: int = 64,
) -> tuple[int, int]:
    """
    This will detect the upscale factor and number of features of a pixelshuffle module in the state dict.

    A pixelshuffle module is a sequence of alternating up convolutions and pixelshuffle.
    The class of this module is commonyl called `Upsample`.
    Examples of such modules can be found in most SISR architectures, such as SwinIR, HAT, RGT, and many more.
    """
    upscale = 1
    num_feat = default_nf

    for i in range(0, 10, 2):
        key = f"{upsample_key}.{i}.weight"
        if key not in state_dict:
            break

        tensor = state_dict[key]
        # we'll assume that the state dict contains tensors
        shape: tuple[int, ...] = tensor.shape  # type: ignore
        num_feat = shape[1]
        upscale *= math.isqrt(shape[0] // num_feat)

    return upscale, num_feat



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict

    in_nc: int = state_dict["tail.1.weight"].shape[0]
    out_nc: int = in_nc

    n_resgroups = get_max_indice(state_dict, "body")
    n_resblocks = get_max_indice(state_dict, "body.0.body")

    max_block_indice: int = get_max_indice(state_dict, "tail.0")
    shape = state_dict[f"tail.0.{max_block_indice}.weight"].shape
    scale = math.isqrt(shape[0] // 3)
    num_feat = shape[1]

    scale, num_feat = get_pixelshuffle_params(state_dict, "tail.0")
    # n_colors = state_dict["tail.1.weight"].shape[0]
    # max_block_indice: int = get_max_indice(state_dict, "feats")
    # scale = math.isqrt(
    #     state_dict[f"tail.0.{max_block_indice}.weight"].shape[0] // 3
    # )

    rgb_range = 255
    kernel_size = state_dict[f"head.0.weight"].shape[-1]
    norm = "sub_mean.weight" in state_dict
    reduction = (
        num_feat // state_dict["body.0.body.0.body.3.conv_du.0.weight"].shape[0]
    )
    res_scale: int = 1


    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RCAN,
        n_resgroups=n_resgroups,
        n_resblocks=n_resblocks,
        n_feats=num_feat,
        reduction=reduction,
        n_colors=in_nc,
        kernel_size=kernel_size,
        res_scale=res_scale,
        rgb_range=rgb_range,
        norm=norm,

        num_feat=num_feat,
        # num_conv=num_conv,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='RCAN',
        detection_keys=(
            "tail.1.weight",
            "body.0.body.0.body.0.weight",
            "body.0.body.0.body.3.conv_du.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

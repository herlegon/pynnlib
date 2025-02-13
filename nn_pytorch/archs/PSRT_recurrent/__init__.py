import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_nsequences,
)
from ..torch_to_onnx import to_onnx
from .module.psrt_recurrent import BasicRecurrentSwin


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 4
    in_nc: int = 3
    out_nc: int = 3

    mid_channels: int = 64
    embed_dim: int = 120
    depths: list[int] = [6, 6, 6]
    num_heads: list[int] = [6, 6, 6]
    window_size: list[int] = [3, 8, 8]
    num_frames: int = 3
    img_size: int = 64
    patch_size: int = 1
    is_low_res_input: bool = True

    mid_channels, embed_dim = state_dict["conv_before_upsample.weight"].shape[:2]
    in_nc = state_dict["conv_first.weight"].shape[1]
    out_nc = in_nc

    nlayers = get_nsequences(state_dict, "patch_align.backward_1.layers")
    depths = []
    for i in range(nlayers):
        depths.append(
            get_nsequences(
                state_dict,
                f"patch_align.backward_1.layers.{i}.residual_group.blocks"
            )
        )


    # We will see later for other values detection...

    # 4126_PSRTRecurrent_mix_precision_REDS_600K_N16
    #   mid_channels: 64
    #   embed_dim: 120
    #   depths: [6, 6, 6]
    #   num_heads: [6,6,6]
    #   window_size: [3, 8, 8]
    #   num_frames: 3
    #   img_size : 64
    #   patch_size : 1
    #   is_low_res_input: True

    # 5123_PSRTRecurrent_mix_precision_Vimeo_300K_N14
    #   mid_channels: 64
    #   embed_dim: 120
    #   depths: [6, 6, 6]
    #   num_heads: [6,6,6]
    #   window_size: [3, 8, 8]
    #   num_frames: 3
    #   is_low_res_input: True

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=BasicRecurrentSwin,
        in_channels=in_nc,
        mid_channels=mid_channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        num_frames=num_frames,
        img_size=img_size,
        patch_size=patch_size,
        cpu_cache_length=0,
        is_low_res_input=is_low_res_input,
        spynet_path="",
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="PSRT recurrent",
        detection_keys=(
            "patch_align.backward_1.layers.0.residual_group.blocks.0.norm1.weight",
            "conv_before_upsample.weight",
            "conv_first.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

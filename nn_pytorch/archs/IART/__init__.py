from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_nsequences,
)
from ..torch_to_onnx import to_onnx
from .module.iart import IART


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 4
    in_nc: int = state_dict["spynet.mean"].shape[1]
    out_nc: int = state_dict["conv_last.weight"].shape[0]

    # fixed
    swin_backbone_nmodules = 4

    mid_channels: int = 64
    embed_dim: int = 120
    depths: list[int] = [6, 6, 6]
    num_heads: list[int] = [6, 6, 6]
    window_size: list[int] = [3, 8, 8]
    num_frames: int = 3
    img_size: int = 64
    patch_size: int = 1
    is_low_res_input: bool = True,

    # swin_backbone_nmodules values:
    use_checkpoint = [True, True, True, True]

    spynet_path: str = ""

    embed_dim = state_dict["conv_before_upsample.weight"].shape[1]
    mid_channels: int = state_dict["upconv1.weight"].shape[0] // 4

    nlayers = get_nsequences(state_dict, "swin_backbone.backward_1.layers")
    depths = []
    for i in range(nlayers):
        depths.append(
            get_nsequences(
                state_dict,
                f"swin_backbone.backward_1.layers.{i}.residual_group.blocks"
            )
        )

    num_heads_num = state_dict[
        "swin_backbone.backward_1.layers.0.residual_group.blocks.1.attn.relative_position_bias_table"
    ].shape[-1]
    num_heads = [num_heads_num for _ in range(nlayers)]

    # We will see later for other values detection...

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=IART,
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
        use_checkpoint=use_checkpoint,
        spynet_path=spynet_path
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="IART",
        detection_keys=(
            "swin_backbone.backward_1.conv_first.weight",
            "swin_backbone.backward_1.layers.0.residual_group.blocks.0.attn.relative_position_bias_table",
            "swin_backbone.forward_1.layers.0.residual_group.blocks.0.norm1.weight",
            "implicit_warp.position_bias",
            "implicit_warp.q.weight",
            "implicit_warp.k.weight",
            "implicit_warp.v.weight",
            "spynet.mean",
            "spynet.basic_module.0.basic_module.0.weight",
            "conv_before_upsample.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

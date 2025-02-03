import math
import re
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ..torch_to_onnx import to_onnx
from .module.SwinIR import SwinIR


def parse(model: PytorchModel) -> None:
    # Defaults
    img_size: int = 64
    in_chans: int = 3
    embed_dim: int = 96
    depths: list[int] = [6, 6, 6, 6]
    num_heads: list[int] = [6, 6, 6, 6]
    window_size: int = 7
    mlp_ratio: float = 4.0
    upscale: int = 2
    img_range: float = 1.0
    upsampler: str = ""
    resi_connection: str = "1conv"
    num_feat: int = 64
    in_nc: int = in_chans
    out_nc: int = in_chans
    start_unshuffle: int = 1

    state_dict = model.state_dict
    state_keys = state_dict.keys()

    if "conv_before_upsample.0.weight" in state_keys:
        if "conv_up1.weight" in state_keys:
            upsampler = "nearest+conv"
        else:
            upsampler = "pixelshuffle"
    elif "upsample.0.weight" in state_keys:
        upsampler = "pixelshuffledirect"

    num_feat_pre_layer = state_dict.get("conv_before_upsample.weight", None)
    num_feat_layer = state_dict.get("conv_before_upsample.0.weight", None)
    num_feat = (
        num_feat_layer.shape[1]
        if num_feat_layer is not None and num_feat_pre_layer is not None
        else 64
    )

    if "conv_first.1.weight" in state_dict:
        state_dict["conv_first.weight"] = state_dict.pop("conv_first.1.weight")
        state_dict["conv_first.bias"] = state_dict.pop("conv_first.1.bias")
        start_unshuffle = round(
            math.sqrt(state_dict["conv_first.weight"].shape[1] // 3)
        )

    in_nc = state_dict["conv_first.weight"].shape[1]
    out_nc = in_nc
    if "conv_last.weight" in state_keys:
        out_nc = state_dict["conv_last.weight"].shape[0]

    upscale = 1
    if upsampler == "nearest+conv":
        upsample_keys = [x for x in state_keys if "conv_up" in x and "bias" not in x]
        for upsample_key in upsample_keys:
            upscale *= 2

    elif upsampler == "pixelshuffle":
        upsample_keys = [
            x
            for x in state_keys
            if "upsample" in x and "conv" not in x and "bias" not in x
        ]
        for upsample_key in upsample_keys:
            shape = state_dict[upsample_key].shape[0]
            upscale *= math.sqrt(shape // num_feat)
        upscale = int(upscale)

    elif upsampler == "pixelshuffledirect":
        upscale = int(math.sqrt(state_dict["upsample.0.bias"].shape[0] // out_nc))

    max_layer_no = 0
    max_block_no = 0
    for key in state_keys:
        if (
            result := re.match(r"layers.(\d*).residual_group.blocks.(\d*).norm1.weight", key)
        ):
            layer_no, block_no = result.groups()
            max_layer_no = max(max_layer_no, int(layer_no))
            max_block_no = max(max_block_no, int(block_no))
    depths = [max_block_no + 1 for _ in range(max_layer_no + 1)]

    if (
        "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        in state_keys
    ):
        num_heads_num = state_dict[
            "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        ].shape[-1]
        num_heads = [num_heads_num for _ in range(max_layer_no + 1)]
    else:
        num_heads = depths

    embed_dim = state_dict["conv_first.weight"].shape[0]

    mlp_ratio = float(
        state_dict["layers.0.residual_group.blocks.0.mlp.fc1.bias"].shape[0] / embed_dim
    )

    # TODO: could actually count the layers, but this should do
    if "layers.0.conv.4.weight" in state_keys:
        resi_connection = "3conv"
    else:
        resi_connection = "1conv"

    window_size = int(
        math.sqrt(
            state_dict[
                "layers.0.residual_group.blocks.0.attn.relative_position_index"
            ].shape[0]
        )
    )

    if "layers.0.residual_group.blocks.1.attn_mask" in state_keys:
        img_size = int(
            math.sqrt(state_dict["layers.0.residual_group.blocks.1.attn_mask"].shape[0])
            * window_size
        )

    # The JPEG models are the only ones with window-size 7, and they also use this range
    img_range = 255.0 if window_size == 7 else 1.0

    model.update(
        arch_name=model.arch.name,
        scale=upscale,
        in_nc=in_nc // start_unshuffle**2,
        out_nc=out_nc,

        ModuleClass=SwinIR,
        in_chans=in_nc,
        num_feat=num_feat,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        img_size=img_size,
        img_range=img_range,
        resi_connection=resi_connection,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='SwinIR',
        detection_keys=(
            "layers.0.residual_group.blocks.0.norm1.weight",
            "conv_first.weight",
            "layers.0.residual_group.blocks.0.mlp.fc1.bias",
            "layers.0.residual_group.blocks.0.attn.relative_position_index",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32'),
    ),
)

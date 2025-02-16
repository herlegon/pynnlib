import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from pynnlib.nn_pytorch.archs.RCAN import get_pixelshuffle_params
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import (
    get_nsequences,
    get_pixelshuffle_params
)
from .module.atd import ATD

def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict

    embed_dim, in_nc = state_dict["conv_first.weight"].shape[:2]
    out_nc: int = in_nc
    window_size: int = math.isqrt(state_dict["relative_position_index_SA"].shape[0])

    num_layers = get_nsequences(state_dict, "layers")
    depths = [6] * num_layers
    num_heads = [6] * num_layers
    for i in range(num_layers):
        depths[i] = get_nsequences(state_dict, f"layers.{i}.residual_group.layers")
        num_heads[i] = state_dict[
            f"layers.{i}.residual_group.layers.0.attn_win.relative_position_bias_table"
        ].shape[1]

    layer0_key: str = "layers.0.residual_group.layers.0"
    num_tokens = state_dict[f"{layer0_key}.attn_atd.scale"].shape[0]
    reducted_dim = state_dict[f"{layer0_key}.attn_atd.wq.weight"].shape[0]
    convffn_kernel_size = state_dict[
        f"{layer0_key}.convffn.dwconv.depthwise_conv.0.weight"
    ].shape[2]
    mlp_ratio: float = state_dict[f"{layer0_key}.convffn.fc1.weight"].shape[0] / embed_dim
    qkv_bias: bool = f"{layer0_key}.wqkv.bias" in state_dict
    ape: bool = bool("absolute_pos_embed" in state_dict)
    patch_norm: bool = bool("patch_embed.norm.weight" in state_dict)

    resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"

    if "conv_up1.weight" in state_dict:
        upsampler = "nearest+conv"
        scale = 4

    elif "conv_before_upsample.0.weight" in state_dict:
        upsampler = "pixelshuffle"
        scale, _ = get_pixelshuffle_params(state_dict, "upsample")

    elif "conv_last.weight" in state_dict:
        upsampler = ""
        scale = 1

    else:
        upsampler = "pixelshuffledirect"
        scale = math.isqrt(state_dict["upsample.0.weight"].shape[0] // in_nc)

    is_light = upsampler == "pixelshuffledirect" and embed_dim == 48
    # use a heuristic for category_size
    category_size: int = 128 if is_light else 256

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=ATD,
        img_size=64,
        patch_size=1,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        category_size=category_size,
        num_tokens=num_tokens,
        reducted_dim=reducted_dim,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        # norm_layer=nn.LayerNorm,
        ape=ape,
        patch_norm=patch_norm,
        img_range=1.,
        upsampler=upsampler,
        resi_connection=resi_connection,
        norm=bool("no_norm" not in state_dict)
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="ATD",
        detection_keys=(
            "relative_position_index_SA",
            "conv_first.weight",
            "conv_first.bias",
            "layers.0.residual_group.layers.0.attn_win.relative_position_bias_table",
            "layers.0.residual_group.layers.0.attn_atd.scale",
            "layers.0.residual_group.layers.0.attn_atd.wq.weight",
            "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight",
            "layers.0.residual_group.layers.0.convffn.fc1.weight",
            "layers.0.residual_group.layers.0.wqkv.bias",
            "layers.0.residual_group.layers.0.attn_aca.logit_scale",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8),
        )
    ),
)

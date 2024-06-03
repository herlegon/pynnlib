import math
from typing import Literal
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx
from .module.dat_arch import DAT




def parse(model: PytorchModel) -> None:
    in_nc: int
    state_dict = model.state_dict
    scale: int = 1

    # rgb_mean = (0.4488, 0.4371, 0.4040)
    # Use these values when trained with NeoSR:
    # rgb_mean = (0.5, 0.5, 0.5)
    img_range = 1.0

    embed_dim, in_nc = state_dict["conv_first.weight"].shape[:2]

    layer_count = get_max_indice(state_dict, "layers") + 1
    depth = [
        get_max_indice(state_dict, f"layers.{i}.blocks") + 1
        for i in range(layer_count)
    ]

    # num_heads is linked to depth
    num_heads = [2] * layer_count
    for i in range(layer_count):
        if depth[i] >= 2:
            # that's the easy path, we can directly read the head count
            num_heads[i] = state_dict[
                f"layers.{i}.blocks.1.attn.temperature"
            ].shape[0]
        else:
            # because of a head_num // 2, we can only reconstruct even head counts
            num_heads[i] = state_dict[
                f"layers.{i}.blocks.0.attn.attns.0.pos.pos3.2.weight"
            ].shape[0] * 2

    upsampler: Literal['pixelshuffle', 'pixelshuffledirect'] = (
        'pixelshuffle'
        if "conv_last.weight" in state_dict
        else 'pixelshuffledirect'
    )
    # Convolutional block before residual connection
    resi_connection: Literal['1conv', '3conv'] = (
        '1conv'
        if "conv_after_body.weight" in state_dict
        else '3conv'
    )

    # Calculate upscale ratio
    if upsampler == 'pixelshuffle':
        scale = 1
        for i in range(0, get_max_indice(state_dict, "upsample") + 1, 2):
            shape, num_feat = state_dict[f"upsample.{i}.weight"].shape[:2]
            scale *= int(math.sqrt(shape // num_feat))
    elif upsampler == 'pixelshuffledirect':
        weight, num_feat = state_dict["upsample.0.weight"].shape[:2]
        scale = int(math.sqrt(weight // in_nc))

    # If True, add a learnable bias to query, key, value.
    qkv_bias: bool = bool("layers.0.blocks.0.attn.qkv.bias" in state_dict)

    # Ratio of ffn hidden dim to embedding dim
    expansion_factor: float = float(
        state_dict["layers.0.blocks.0.ffn.fc1.weight"].shape[0]
        / embed_dim
    )

    # Input image size
    if "layers.0.blocks.2.attn.attn_mask_0" in state_dict:
        img_size = int(math.sqrt(
            math.prod(
                state_dict["layers.0.blocks.2.attn.attn_mask_0"].shape[:2]
            )
        ))

    # Height and Width of spatial window.
    if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_dict:
        split_sizes = (
            state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
        )
        split_size = [int(x) for x in split_sizes]

    arch_name: str = model.arch.name
    if len(depth) == 1:
        arch_subtype = "_light"
    else:
        if split_size == [8, 16]:
            arch_subtype = "_small"
        elif expansion_factor == 4:
            arch_subtype = "_medium"
        elif expansion_factor == 2:
            arch_subtype = "2"
    arch_name = f"{model.arch.name}{arch_subtype}"

    # Update model parameters
    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=in_nc,

        ModuleClass=DAT,
        img_size=img_size,
        in_chans=in_nc,
        embed_dim=embed_dim,
        split_size=split_size,
        depth=depth,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        qkv_bias=qkv_bias,
        upscale=scale,
        img_range=img_range,
        resi_connection=resi_connection,
        upsampler=upsampler,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='DAT',
        detection_keys=(
            "conv_first.weight",
            "layers.0.blocks.0.ffn.fc1.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=['fp32', 'bf16'],
    ),
)

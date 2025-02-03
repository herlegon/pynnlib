import torch
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...inference.session import PyTorchSession
from ..helpers import get_scale_and_out_nc
from ..torch_to_onnx import to_onnx
from .module.span import SPAN


def parse(model: PytorchModel) -> None:
    in_nc: int = 3
    state_dict = model.state_dict

    feature_channels, in_nc = state_dict["conv_1.sk.weight"].shape[:2]
    scale, out_nc = get_scale_and_out_nc(
        state_dict["upsampler.0.weight"].shape[0],
        in_nc,
    )

    eval_conv_value = None
    try:
        eval_conv_value = state_dict["conv_1.eval_conv.weight"]
    except:
        pass
    eval_conv: bool = bool(eval_conv_value is not None)

    # Values from original source code
    norm: bool = True
    img_range: float = 255.0
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040)
    if "no_norm" in state_dict:
        state_dict["no_norm"] = torch.zeros(1)
        norm = False
        img_range = 1.0,
        rgb_mean = (0.5, 0.5, 0.5)

    # Update model parameters
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=SPAN,
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        upscale=scale,
        feature_channels=feature_channels,
        eval_conv=eval_conv,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='SPAN',
        detection_keys=(
            "conv_1.sk.weight",
            "block_1.c1_r.sk.weight",
            "block_1.c1_r.eval_conv.weight",
            "block_1.c3_r.eval_conv.weight",
            "conv_cat.weight",
            "conv_2.sk.weight",
            "conv_2.eval_conv.weight",
            "upsampler.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
    ),
)

import math
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx
from .module.realplksr_arch import RealPLKSR


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    dim: int = state_dict["feats.0.weight"].shape[0]
    max_block_indice: int = get_max_indice(state_dict, "feats")
    n_blocks: int = max_block_indice - 2
    scale = math.isqrt(
        state_dict[f"feats.{max_block_indice}.weight"].shape[0] // 3
    )

    shape: str = state_dict["feats.1.lk.conv.weight"].shape
    kernel_size, split_ratio = shape[2], shape[0] / dim

    use_ea: bool = bool("feats.1.attn.f.0.weight" in state_dict)

    use_dysample = "to_img.init_pos" in state_dict

    model.update(
        arch_name=f"{model.arch_name} DySample" if use_dysample else model.arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RealPLKSR,
        dim=dim,
        n_blocks=n_blocks,
        upscaling_factor=scale,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        use_ea=use_ea,
        # norm_groups: int = 4,
        # dropout: float = 0,
        dysample=use_dysample,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RealPLKSR",
        detection_keys=(
            "feats.0.weight",
            "feats.1.channel_mixer.0.weight",
            "feats.1.lk.conv.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16', 'bf16']),
    ),
)

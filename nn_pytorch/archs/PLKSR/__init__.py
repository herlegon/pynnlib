import math
from typing import Literal
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx
from .module.plksr_arch import PLKSR


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    dim: int = state_dict["feats.0.weight"].shape[0]
    max_block_indice: int = get_max_indice(state_dict, "feats")
    n_blocks: int = max_block_indice - 1
    scale = math.isqrt(
        state_dict[f"feats.{max_block_indice}.weight"].shape[0] // 3
    )

    ccm_0_shape = state_dict["feats.1.channe_mixer.0.weight"].shape[2]
    ccm_2_shape = state_dict["feats.1.channe_mixer.2.weight"].shape[2]
    if ccm_0_shape == 3 and ccm_0_shape == 1:
        ccm_type = "CCM"
    elif ccm_0_shape == 1 and ccm_0_shape == 3:
        ccm_type = "ICCM"
    elif ccm_0_shape == 3 and ccm_2_shape == 3:
        ccm_type = "DCCM"

    lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK"
    kernel_size: int = 17
    key: str = ""
    if "feats.1.lk.conv.weight" in state_dict:
        # PLKConv2d
        lk_type, key = "PLK", "feats.1.lk.conv.weight"
    elif "feats.1.lk.mn_conv.weight" in state_dict:
        # RectSparsePLKConv2d
        lk_type, key = "RectSparsePLK", "feats.1.lk.mn_conv.weight"
    elif "feats.1.lk.convs.0.weight" in state_dict:
        # SparsePLKConv2d
        lk_type, key = "SparsePLK", "feats.1.lk.convs.0.weight"
    else:
        raise ValueError("Unknown LK type")
    shape = state_dict[key].shape
    kernel_size, split_ratio = shape[2], shape[0]/dim
    if lk_type == "SparsePLKConv2d":
        kernel_size = 17

    use_ea: bool = bool("feats.1.attn.f.0.weight" in state_dict)

    # Update model parameters
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=PLKSR,
        dim=dim,
        n_blocks=n_blocks,
        upscaling_factor=scale,
        ccm_type=ccm_type,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        lk_type=lk_type,
        use_max_kernel=False,
        # sparse_kernels: Sequence[int] = [5, 5, 5, 5],
        # sparse_dilations: Sequence[int] = [1, 2, 3, 4],
        # with_idt: bool = False,
        use_ea=use_ea
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="PLKSR",
        detection_keys=(
            "feats.0.weight",
            "feats.1.channe_mixer.0.weight",
            "feats.1.channe_mixer.2.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16', 'bf16']),
    ),
)

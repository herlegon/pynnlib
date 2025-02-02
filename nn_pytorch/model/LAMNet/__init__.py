import math
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import get_max_indice
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from .module.lamnet import LAMNet
from .module.fsa_func import compile_fsa_ext


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    out_nc: int = 3
    in_nc: int = out_nc

    compile_fsa_ext()

    scale: int = int(math.sqrt(state_dict["upsample.0.weight"].shape[0] // 3))
    dim: int = state_dict["upsample.0.weight"].shape[1]
    num_groups: int = get_max_indice(state_dict, "deep_feature_extraction")

    # fixed because I don't know how to detect that yet and i don't care
    num_blocks: int = 6
    kernel_size: int = 13
    kernel_loc: tuple[int] = (4, 6, 7)
    kernel_stride: tuple[int] = (1, 2, 4)
    num_head: int = 4
    expansion_factor: float = 1.0
    rgb_mean: tuple[float] = (0.4488, 0.4371, 0.4040)

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=LAMNet,
        num_blocks=num_blocks,
        num_groups=num_groups,
        dim=dim,
        kernel_size=kernel_size,
        kernel_loc=kernel_loc,
        kernel_stride=kernel_stride,
        num_head=num_head,
        expansion_factor=expansion_factor,
        rgb_mean=rgb_mean,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="LAMNet large",
        detection_keys=(
            "deep_feature_extraction.5.weight",
            "deep_feature_extraction.0.alpha",
            "upsample.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
    NnPytorchArchitecture(
        name="LAMNet",
        detection_keys=(
            "deep_feature_extraction.4.weight",
            "deep_feature_extraction.0.alpha",
            "upsample.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)

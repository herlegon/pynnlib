from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
)
from .module.vsr_arch import MSRSWVSR
from ..torch_to_onnx import to_onnx


def parse(model: PytorchModel) -> None:
    num_feat: int = 64
    in_nc: int = 3
    num_block: tuple[int, int, int] = (5, 3, 2)
    scale: int = 4

    # max_indice = get_max_indice(model.state_dict, "body")
    in_nc: int = 3
    # num_feat, in_nc = model.state_dict["body.0.weight"].shape[:2]
    # num_conv: int = (max_indice - 2) // 2
    pixelshuffle_shape: int = model.state_dict[f"body.{max_indice}.bias"].shape[0]
    scale, out_nc = get_scale_and_out_nc(pixelshuffle_shape, in_nc)

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        num_feat=num_feat,
        # num_conv=num_conv,

        ModuleClass=MSRSWVSR,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='AnimeSR',
        detection_keys=(
            "non_implemented",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        is_temporal=True,
        # size_constraint=SizeConstraint(
        #     min=(64, 64)
        # )
    ),
)

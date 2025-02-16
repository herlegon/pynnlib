from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
)
from .module.SRVGG import SRVGGNetCompact
from ..torch_to_onnx import to_onnx


def parse(model: PytorchModel) -> None:
    max_indice = get_max_indice(model.state_dict, "body")

    num_feat, in_nc = model.state_dict["body.0.weight"].shape[:2]
    num_conv: int = (max_indice - 2) // 2
    pixelshuffle_shape: int = model.state_dict[f"body.{max_indice}.bias"].shape[0]
    scale, out_nc = get_scale_and_out_nc(pixelshuffle_shape, in_nc)

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=SRVGGNetCompact,
        num_feat=num_feat,
        num_conv=num_conv,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RealESRGAN (Compact)",
        detection_keys=(
            "body.0.weight",
            "body.1.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

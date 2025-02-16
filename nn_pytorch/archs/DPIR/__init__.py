from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.network_unet import NonLocalUNet, UNetRes


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    out_nc: int = state_dict["m_tail.weight"].shape[0]
    in_nc: int = out_nc

    nc: list[int] = list([
        state_dict[f"m_down{i+1}.0.res.0.weight"].shape[0] for i in range(3)
    ])
    nc.append(state_dict[f"m_body.0.res.0.weight"].shape[0])

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=UNetRes,
        nc=nc,
        nb=4,
        act_mode='R',
        downsample_mode='strideconv',
        upsample_mode='convtranspose',
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="DPIR",
        detection_keys=(
            "m_head.weight",
            "m_down1.0.res.0.weight",
            "m_down2.0.res.0.weight",
            "m_down3.0.res.0.weight",
            "m_body.0.res.0.weight",
            "m_tail.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)

from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.network_scunet import SCUNet


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    dim, in_nc = state_dict["m_head.0.weight"].shape[:2]
    config = (
        get_max_indice(state_dict, "m_down1"),
        get_max_indice(state_dict, "m_down2"),
        get_max_indice(state_dict, "m_down3"),
        get_max_indice(state_dict, "m_body") + 1,
        get_max_indice(state_dict, "m_up3"),
        get_max_indice(state_dict, "m_up2"),
        get_max_indice(state_dict, "m_up1")
    )

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=in_nc,

        ModuleClass=SCUNet,
        dim=dim,
        config=config,
        drop_path_rate=0,
        input_resolution=256
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='SCUNet',
        detection_keys=(
            "m_head.0.weight",
            "m_tail.0.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

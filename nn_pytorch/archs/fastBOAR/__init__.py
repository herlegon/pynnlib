from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.fast_two_step_boar import ResUNet


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    dim, in_nc = state_dict["m_head.0.weight"].shape[:2]
    scale: int = 1


    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=in_nc,

        ModuleClass=ResUNet,
        dim=dim,
        # config=config,
        # drop_path_rate=0,
        # input_resolution=256
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='FastBOAR',
        detection_keys=(
            "m_head.weight",
            "m_down1.0.res.0.weight",
            "m_body.0.res.0.weight",
            "m_body.0.res.2.weight"
            "m_up1.2.res.2.weight",
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

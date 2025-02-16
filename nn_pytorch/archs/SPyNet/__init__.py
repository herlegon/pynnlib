from pynnlib.architecture import (
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PytorchModel
from .module.SPyNet import SpyNet


def parse(model: PytorchModel) -> None:
    # state_dict: StateDict = model.state_dict
    scale: int = 1
    in_nc: int = 3
    out_nc: int = 3

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=SpyNet,
        # return_levels=[5],
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="SPyNet",
        detection_keys=(
            "basic_module.0.basic_module.0.weight",
            "basic_module.3.basic_module.0.weight",
            "basic_module.5.basic_module.0.weight",
        ),
        parse=parse,
        # to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)

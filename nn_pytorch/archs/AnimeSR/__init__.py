import math
from pynnlib.architecture import (
    InferType,
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PytorchModel
from utils.p_print import *
from .module.vsr_arch import MSRSWVSR
from ..torch_to_onnx import to_onnx


def parse(model: PytorchModel) -> None:
    recurrent_cell_shape = model.state_dict[f"recurrent_cell.conv_s1_first.0.weight"].shape
    print(recurrent_cell_shape)
    in_nc: int = recurrent_cell_shape[3]
    num_feat: int = recurrent_cell_shape[0]
    scale: int = int(
        math.sqrt((recurrent_cell_shape[1] - num_feat - in_nc * 3) // in_nc)
    )
    num_block: tuple[int, int, int] = (5, 3, 2)
    out_nc: int = in_nc

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        num_feat=num_feat,
        num_block=num_block,

        ModuleClass=MSRSWVSR,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='AnimeSR',
        detection_keys=(
            "recurrent_cell.conv_s1_first.0.weight",
            "recurrent_cell.fusion.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        # size_constraint=SizeConstraint(
        #     min=(64, 64)
        # )
        infer_type=InferType(
            type='temporal',
            inputs=3,
            outputs=1,
        )
    ),
)

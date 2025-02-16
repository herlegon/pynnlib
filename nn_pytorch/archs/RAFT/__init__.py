import re
from pynnlib.architecture import (
    InferType,
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from .module.raft import RAFT



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 1
    in_nc: int = 3
    out_nc: int = 3

    num_feat, in_nc = state_dict["fnet.conv1.weight"].shape[:2]

    arch_name: str = model.arch.name
    is_small: bool = False
    if num_feat < 64:
        is_small = True
        arch_name += " (small)"

    # shape = state_dict["update_block.flow_head.conv1.weight"].shape
    # hdim: int = shape[1]

    # let's calculate like that even if not sure it's like that, i don't care
    corr_radius: int = (
        state_dict["update_block.encoder.convc1.bias"].shape[0]
        // state_dict["update_block.encoder.convf2.bias"].shape[0]
    )
    corr_levels: int = 4

    model.update(
        arch_name=arch_name,
        scale=scale,
        # not applicable but
        in_nc=in_nc,
        out_nc=out_nc,
        num_feat=num_feat,

        ModuleClass=RAFT,
        # corr_levels=corr_levels,
        # corr_radius=corr_radius,
        # dropout=0
        # alternate_corr=is_compiled
        small=is_small
        # mixed_precision=False,
        # hdim=hdim,
        # cdim=cdim,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RAFT",
        detection_keys=(
            "fnet.conv1.weight",
            "update_block.flow_head.conv1.weight",
            "update_block.flow_head.conv1.bias",

        ),
        parse=parse,
        # to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        ),
        infer_type=InferType(
            type='optical_flow',
            inputs=4,
            outputs=1,
        )
    ),
)

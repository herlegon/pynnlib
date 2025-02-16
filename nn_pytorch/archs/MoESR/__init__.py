from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx
from .module.MoESR import MoESR


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    upsample = ['conv', 'pixelshuffledirect', 'pixelshuffle', 'nearest+conv', 'dysample']
    dim, in_nc = state_dict['in_to_dim.weight'].shape[:2]
    n_blocks = get_nsequences(state_dict, 'blocks')
    n_block = get_nsequences(state_dict, 'blocks.0.blocks')
    expansion_factor_shape = state_dict['blocks.0.blocks.0.fc1.weight'].shape
    expansion_factor = (expansion_factor_shape[0] / expansion_factor_shape[1]) / 2
    expansion_msg_shape = state_dict['blocks.0.msg.gated.0.fc1.weight'].shape
    expansion_msg = (expansion_msg_shape[0] / expansion_msg_shape[1]) / 2
    _, index, scale, _, out_nc, upsample_dim, _ = state_dict['upscale.MetaUpsample']
    upsampler = upsample[int(index)]

    # Update model parameters
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=MoESR,
        n_blocks=n_blocks,
        n_block=n_block,
        dim=dim,
        expansion_factor=expansion_factor,
        expansion_msg=expansion_msg,
        upsampler=upsampler,
        upsample_dim=int(upsample_dim),
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="MoESR",
        detection_keys=(
            'in_to_dim.weight',
            'in_to_dim.bias',
            'blocks.0.blocks.0.gamma',
            'blocks.0.blocks.0.norm.weight',
            'blocks.0.blocks.0.norm.bias',
            'blocks.0.blocks.0.fc1.weight',
            'blocks.0.blocks.0.fc1.bias',
            'blocks.0.blocks.0.conv.dwconv_hw.weight',
            'blocks.0.blocks.0.conv.dwconv_hw.bias',
            'blocks.0.blocks.0.conv.dwconv_w.weight',
            'blocks.0.blocks.0.conv.dwconv_w.bias',
            'blocks.0.blocks.0.conv.dwconv_h.weight',
            'blocks.0.blocks.0.conv.dwconv_h.bias',
            'blocks.0.blocks.0.fc2.weight',
            'blocks.0.blocks.0.fc2.bias',
            'upscale.MetaUpsample',
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16']),
    ),
)

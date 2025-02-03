import json
import os
from pprint import pprint
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.xrestormer import XRestormer


def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    dim, in_nc = state_dict["patch_embed.proj.weight"].shape[:2]
    out_nc: int = in_nc

    # not possible to detect scale. model has to be modified ?
    # or ignore sr, or use as param

    # denoise
    # deblur
    # derain
    # dehaze
    # -> scale = 1

    # sr
    # -> scale = 4

    scale: int = 1

    # use filename, I f*cking hate that
    if "sr" in os.path.splitext(os.path.basename(model.filepath))[0]:
        scale = 4

    # for the other params, I don't want to detect them unless it's a good arch/model
    # don't want to train a model
    # use params defined in repo
    # channel_heads: use channel_attn.temperature

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=XRestormer,
        dim=dim,
        num_blocks=(2, 4, 4, 4),
        num_refinement_blocks=4,
        channel_heads=(1, 2, 4, 8),
        spatial_heads=(1, 2, 4, 8),
        overlap_ratio=(0.5, 0.5, 0.5, 0.5),
        window_size=8,
        spatial_dim_head=16,
        bias=False,
        ffn_expansion_factor=2.66,
        layer_norm_type='WithBias',
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='X-Restormer',
        detection_keys=(
            "patch_embed.proj.weight",
            "encoder_level1.0.spatial_attn.project_out.weight",
            "output.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)

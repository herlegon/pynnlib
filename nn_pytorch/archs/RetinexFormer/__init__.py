from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.RetinexFormer import RetinexFormer

def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 1

    out_nc: int = 3
    if "conv_out.weight" in state_dict:
        out_nc = state_dict["conv_out.weight"].shape[0]
    else:
        out_nc = state_dict["body.0.denoiser.mapping.weight"].shape[0]

    if "conv_in.weight" in state_dict:
        weight = state_dict["conv_in.weight"]
    else:
        weight = state_dict["body.0.denoiser.embedding.weight"]
    n_feat, in_nc = weight.shape[:2]

    stage = get_max_indice(state_dict, "body") + 1

    s: str = ""
    if "body.0.denoiser.encoder_layers.0.0.blocks.0.0.rescale" in state_dict:
        s = "denoiser"
    num_blocks = [
        get_max_indice(state_dict, f"body.0.{s}.encoder_layers.0.0.blocks") + 1,
        get_max_indice(state_dict, f"body.0.{s}.encoder_layers.1.0.blocks") + 1,
        get_max_indice(state_dict, f"body.0.{s}.bottleneck.blocks") + 1,
    ]

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RetinexFormer,
        n_feat=n_feat,
        stage=stage,
        num_blocks=num_blocks,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='RetinexFormer',
        detection_keys=(
            "body.0.denoiser.mapping.weight",
            "body.0.denoiser.embedding.weight",
            "body.0.denoiser.bottleneck.blocks.0.0.rescale",
            "body.0.denoiser.encoder_layers.0.0.blocks.0.0.rescale",
            "body.0.denoiser.encoder_layers.1.0.blocks.0.0.rescale",
            "body.0.denoiser.encoder_layers.1.1.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8),
            max=(3000, 3000),
        )
    ),
    # NnPytorchArchitecture(
    #     name='MST++',
    #     detection_keys=(
    #         "body.2.mapping.weight",
    #         "conv_in.weight",
    #         "body.0.bottleneck.blocks.0.0.rescale",
    #         "body.0.encoder_layers.0.0.blocks.0.0.rescale",
    #         "body.0.encoder_layers.1.0.blocks.0.0.rescale",
    #         "body.0.encoder_layers.1.1.weight",
    #     ),
    #     parse=parse,
    #     to_onnx=to_onnx,
    #     dtypes=('fp32', 'fp16'),
    #     size_constraint=SizeConstraint(
    #         min=(8, 8),
    #         max=(3000, 3000),
    #     )
    # ),
)

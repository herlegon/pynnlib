from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel

from .module.network_rvrt import RVRT
from ..torch_to_onnx import to_onnx
from ...torch_types import StateDict



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    if "feat_extract.1.weight" in state_dict:
        in_nc: int = state_dict["feat_extract.1.weight"].shape[1]
    else:
        in_nc: int = state_dict["feat_extract.main.1.weight"].shape[1]

    out_nc: int = state_dict["conv_last.weight"].shape[0]

    # I don't care detection all params until quality is proven to be good
    # clip_size = 2
    # embed_dims=
        # [
        #     feat_extract.1.bias[0]
        # deform_align.backward_1.proj_q.1.weight
        #     reconstruction.main.1.bias[0]
        # ]
    # img_size = feat_extract.6.main.5.0.residual_group.blocks.0.attn.relative_position_index

    #   window_size = [2, 8, 8]
    #   num_blocks=[1, 2, 1],
    #   depths=[2, 2, 2],
    #   num_heads=[6, 6, 6],

    #   inputconv_groups[0] = feat_extract.1.weight[0] ?
    #   inputconv_groups[1] = feat_extract.1.weight[1] ?

    #   deformable_groups=12
    #   attention_heads=12
    #       deform_align
    #   attention_window=[3, 3],
    #       # deform_align.backward_1.conv_offset.2.weight ?

    scale: int = 4

    clip_size: int = 2
    img_size: tuple[int, int, int] = (2, 64, 64)
    window_size: tuple[int, int, int] = (2, 8, 8)
    # RSTB blocks
    num_blocks: tuple[int, int, int] = (1, 2, 1)
    depths: tuple[int, int, int] = (2, 2, 2)
    embed_dims: tuple[int, int, int] = (144, 144, 144)
    num_heads: tuple[int, int, int] = (6, 6, 6)
    mlp_ratio: float = 2.
    qkv_bias: bool = True,
    # norm_layer: nn.Module = nn.LayerNorm
    inputconv_groups: tuple[int] = (1, 1, 1, 1, 1, 1)
    spynet_path="iuehfiuzefhiuyz"
    deformable_groups: int = 12
    attention_heads: int = 12
    attention_window: tuple[int, int] = (3, 3)
    nonblind_denoising: bool = False

    # Denoising model
    if "feat_extract.1.weight" in state_dict:
        scale = 1
        shape = state_dict["feat_extract.1.weight"].shape
        nonblind_denoising: bool = True if shape[1] == in_nc + 1 else False
        if nonblind_denoising:
            inputconv_groups = (1, 3, 4, 6, 8, 4)
        else:
            inputconv_groups = (1, 3, 3, 3, 3, 3)

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RVRT,
        clip_size=clip_size,
        img_size=img_size,
        window_size=window_size,
        num_blocks=num_blocks,
        depths=depths,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        # norm_layer=norm_layer,
        inputconv_groups=inputconv_groups,
        spynet_path=spynet_path,
        deformable_groups=deformable_groups,
        attention_heads=attention_heads,
        attention_window=attention_window,
        nonblind_denoising=nonblind_denoising,
    )

    # 001_RVRT_videosr_bi_REDS_30frames
        # upscale=4,
        # clip_size=2,
        # img_size=[2, 64, 64],
        # window_size=[2, 8, 8],
        # num_blocks=[1, 2, 1],
        # depths=[2, 2, 2],
        # embed_dims=[144, 144, 144],
        # num_heads=[6, 6, 6],
        # inputconv_groups=[1, 1, 1, 1, 1, 1],
        # deformable_groups=12, attention_heads=12,
        # attention_window=[3, 3],
        # nonblind_denoising = False

    # 002_RVRT_videosr_bi_Vimeo_14frames
    # 003_RVRT_videosr_bd_Vimeo_14frames
        # upscale=4,
        # clip_size=2,
        # img_size=[2, 64, 64],
        # window_size=[2, 8, 8],
        # num_blocks=[1, 2, 1],
        # depths=[2, 2, 2],
        # embed_dims=[144, 144, 144],
        # num_heads=[6, 6, 6],
        # inputconv_groups=[1, 1, 1, 1, 1, 1],
        # deformable_groups=12, attention_heads=12,
        # attention_window=[3, 3],
        # nonblind_denoising = False

    # 004_RVRT_videodeblurring_DVD_16frames
        # upscale=1,
        # clip_size=2,
        # img_size=[2, 64, 64],
        # window_size=[2, 8, 8],
        # num_blocks=[1, 2, 1],
        # depths=[2, 2, 2],
        # embed_dims=[192, 192, 192],
        # num_heads=[6, 6, 6],
        # inputconv_groups=[1, 3, 3, 3, 3, 3],
        # deformable_groups=12,
        # attention_heads=12,
        # attention_window=[3, 3],
        # nonblind_denoising = False

    # 005_RVRT_videodeblurring_GoPro_16frames
        # upscale=1,
        # clip_size=2,
        # img_size=[2, 64, 64],
        # window_size=[2, 8, 8],
        # num_blocks=[1, 2, 1],
        # depths=[2, 2, 2],
        # embed_dims=[192, 192, 192],
        # num_heads=[6, 6, 6],
        # inputconv_groups=[1, 3, 3, 3, 3, 3],
        # deformable_groups=12,
        # attention_heads=12,
        # attention_window=[3, 3],
        # nonblind_denoising = False

    # 006_RVRT_videodenoising_DAVIS_16frames
        # upscale=1,
        # clip_size=2,
        # img_size=[2, 64, 64],
        # window_size=[2, 8, 8],
        # num_blocks=[1, 2, 1],
        # depths=[2, 2, 2],
        # embed_dims=[192, 192, 192],
        # num_heads=[6, 6, 6],
        # inputconv_groups=[1, 3, 4, 6, 8, 4],
        # deformable_groups=12,
        # attention_heads=12,
        # attention_window=[3, 3],
        # nonblind_denoising = TRUE!!!<



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='RVRT',
        detection_keys=(
            "spynet.basic_module.0.basic_module.0.weight",
            "deform_align.backward_1.proj_q.1.weight",
            "deform_align.backward_1.conv_offset.2.weight",
            "reconstruction.main.1.bias",
            "conv_last.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

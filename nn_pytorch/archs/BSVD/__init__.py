
import math
from pynnlib.architecture import (
    InferType,
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PytorchModel
from utils.p_print import *
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from .module.bsvd import BSVD
from .module.tsm import TSN



def _parse_tsn(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    in_nc: int = 3
    out_nc: int = in_nc
    scale: int = 1


    # dump_0407_test_set8_DAVIS_0402_train_DenoisingNet_tsm_temp11_none_b8_dp_g07_s50k_2gpu_blind_c64
    # network_g:
    #   type: TSN
    #   num_segments: 11
    #   modality: 'RGB'
    #   base_model: WNet_multistage
    #   consensus_type: 'avg'
    #   dropout: 0
    #   img_feature_dim: 256
    #   partial_bn: True
    #   pretrain: 'imagenet'
    #   shift_type: TSM
    #   shift_div: 8
    #   shift_place: blockres
    #   fc_lr5: True
    #   temporal_pool: False
    #   non_local: False
    #   unet_kernel_size: 3
    #   max_pts_stride: 1
    #   net2d_opt:
    #     chns: [64, 128, 256]
    #     mid_ch: 64
    #     shift_input: False
    #     norm: 'none'
    #     blind: True
    #   inplace: False


    # 0328_test_set8_DAVIS_0224_DenoisingNet_tsm_toFutureOnly_temp11_none_b8_dp_g07_s50k_2gpu
    # network_g:
    #   type: TSN
    #   num_segments: 11
    #   modality: 'RGB'
    #   base_model: WNet_multistage
    #   consensus_type: 'avg'
    #   dropout: 0
    #   img_feature_dim: 256
    #   partial_bn: True
    #   pretrain: 'imagenet'
    #   shift_type: TSM_toFutureOnly
    #   shift_div: 8
    #   shift_place: blockres
    #   fc_lr5: True
    #   temporal_pool: False
    #   non_local: False
    #   unet_kernel_size: 3
    #   max_pts_stride: 1
    #   net2d_opt:
    #     chns: [32, 64, 128]
    #     mid_ch: 32
    #     shift_input: False
    #     norm: 'none'
    #   inplace: False

    in_nc = state_dict["base_model.nets_list.0.inc.convblock.0.weight"].shape[1]
    num_segments = 11
    shift_div = 8

    model.update(
        arch_name="TSN",
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=TSN,
        num_segments=num_segments,
        base_model="WNet_multistage",
        shift_type="TSM",
        shift_div=shift_div,
        inplace=False,
        net2d_opt={},
        enable_past_buffer=True
    )



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    in_nc: int = 3
    out_nc: int = in_nc
    scale: int = 1

    interm_ch, in_ch = state_dict["base_model.nets_list.0.inc.convblock.0.weight"].shape[:2]
    if in_ch == 3:
        _parse_tsn(model=model)
        return

    # bsvd_c64
    #   type: BSVD
    #   chns: [64, 128, 256]
    #   mid_ch: 64
    #   shift_input: False
    #   norm: 'none'
    #   interm_ch: 64
    #   act: 'relu6'
    #   pretrain_ckpt: './experiments/pretrained_ckpt/bsvd-64.pth'

    in_nc = in_ch - 1
    out_nc = state_dict["base_model.nets_list.1.outc.convblock.3.weight"].shape[0]

    chns: list[int] = [
        state_dict["base_model.nets_list.0.downc0.convblock.0.weight"].shape[1],
        state_dict["base_model.nets_list.0.downc1.convblock.0.weight"].shape[1],
        state_dict["base_model.nets_list.0.upc2.convblock.0.c1.net.weight"].shape[1],
    ]

    mid_ch: int = 64
    # mid_ch: "base_model.nets_list.0.inc.convblock.3.weight" ?

    # shift_input = True if CvBlock else InputCvBlock
    shift_input: bool
    if "base_model.nets_list.0.inc.convblock.3.weight" in state_dict:
        shift_input = False
    else:
        shift_input = True

    norm: str = "none"
    act: str = "relu6"
    blind: bool = False

    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_ch - 1,
        out_nc=out_nc,

        ModuleClass=BSVD,
        chns=chns,
        mid_ch=mid_ch,
        shift_input=shift_input,
        in_ch=in_ch,
        out_ch=out_nc,
        norm=norm,
        act=act,
        interm_ch=interm_ch,
        blind=blind,
        pretrain_ckpt=None
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='BSVD',
        # !!! currently no way to distingate BSVD and TSN except from shapes
        detection_keys=(
            "base_model.nets_list.1.outc.convblock.3.weight",
            "base_model.nets_list.0.inc.convblock.0.weight",
            "base_model.nets_list.1.inc.convblock.0.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        # size_constraint=SizeConstraint(
        #     min=(64, 64)
        # )
        infer_type=InferType(
            type='temporal',
        )
    ),
)



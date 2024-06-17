from pynnlib.utils.p_print import *
from pynnlib.utils import path_split
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from .module.upcunet import UpCunet


_shape_to_scale: dict[int, list[int]] = {
    2: [ 64, 3, 4, 4],
    3: [ 64, 3, 5, 5],
    4: [ 64, 64, 4, 4],
}

_BASENAME_TO_PARAMS: dict[str, tuple[bool|int]] = {
    "up2x-latest-conservative": [False, 2, -1],
    "up2x-latest-no-denoise": [False, 2, 0],
    "up2x-latest-denoise1x": [False, 2, 1],
    "up2x-latest-denoise2x": [False, 2, 2],
    "up2x-latest-denoise3x": [False, 2, 3],
    "up3x-latest-conservative": [False, 3, -1],
    "up3x-latest-no-denoise": [False, 3, 0],
    "up4x-latest-conservative": [False, 4, -1],
    "up4x-latest-no-denoise": [False, 4, 0],
    "up4x-latest-denoise3x": [False, 4, 3],
    "pro-conservative-up2x": [True, 2, -1],
    "pro-no-denoise-up2x": [True, 2, 0],
    "pro-denoise3x-up2x": [True, 2, 3],
    "pro-conservative-up3x": [True, 3, -1],
    "pro-no-denoise-up3x": [True, 3, 0],
    "pro-denoise3x-up3x": [True, 3, 3],
}
_PARAMS_TO_BASENAME: dict[str, str] = dict([('_'.join([str(x) for x in v]), k) for k, v in _BASENAME_TO_PARAMS.items()])


def basename_to_params(basename: str) -> tuple[bool, int, int]:
    # Let's consider latest trained model by the community
    # are trained with the 'pro' implementation
    pro, scale, denoise = True, 2, -1
    basename = basename.lower()
    try:
       pro, scale, denoise =_BASENAME_TO_PARAMS[basename]
    except:
        pass

    return pro, scale, denoise


def params_to_basename(pro: bool=True, scale: int=2, denoise: int=-1) -> str:
    """Returns basename, args:
    scale: 2/3/4 (default:2\n
    pro: True, False (default:True)\n
    denoise: -1/0/1/2/3 (default:-1)
    """
    k = f'{pro}_{scale}_{denoise}'
    basename = ''
    try:
        basename = _PARAMS_TO_BASENAME[k]
    except:
        raise ValueError(f"No model found for pro={pro}, scale={scale}, denoise={denoise}")

    return basename



def parse(model: PytorchModel) -> None:
    state_dict = model.state_dict

    in_nc: int = state_dict["unet1.conv1.conv.0.weight"].shape[1]

    scale: int = 0
    shape: list[int] = list(state_dict["unet1.conv_bottom.weight"].shape)
    for s, v in _shape_to_scale.items():
        if shape == v:
            scale = s
            break

    out_nc: int = state_dict["unet2.conv_bottom.weight"].shape[0]
    shuffle_factor: int = 0
    if scale == 4:
        shuffle_factor = 2
        out_nc = 3

    # Let's consider latest trained models are 'pro'
    pro = True
    denoise = 0
    if StateDict.get(state_dict, 'metadata.by', None) == 'herlegon':
        pro = state_dict["metadata.pro"]
        denoise = state_dict["metadata.denoise"]
    else:
        pro = True if StateDict.get(state_dict, 'pro', 0) == 1 else False
        try:
            del state_dict["pro"]
        except:
            pass

        # How I became stupid: use basename! arghhhhh... f.o.!
        pro, _, denoise = basename_to_params(path_split(model.filepath)[1])

    model.update(
        arch_name=model.arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=UpCunet,
        legacy=not pro,
        shuffle_factor=shuffle_factor,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name='Real_CUGAN',
        detection_keys=(
            "unet1.conv1.conv.0.weight",
            "unet1.conv_bottom.weight",
            "unet2.conv_bottom.weight"
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)

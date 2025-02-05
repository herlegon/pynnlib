from __future__ import annotations
from collections import OrderedDict
import glob
import inspect
import math
from torch import Tensor
from pynnlib.model import PytorchModel
from ..torch_types import StateDict



def find_compiler_bindir() -> str | None:
    patterns = (
        "C:\\Visual_Studio\\*\\Community\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64",
        "C:\\Visual_Studio *\\vc/bin",
        "C:\\Program Files\\Microsoft Visual Studio\\*\\Community\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64",
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None



def get_max_indice(state_dict: StateDict, seq_key: str) -> int:
    # start_time = time.time()
    # for _ in range(10**4):
    prefix = f"{seq_key}."
    prefix_len: int = len(prefix)
    keys: list[int] = [
        int(k[prefix_len: ].split(".", maxsplit=1)[0])
        for k in state_dict.keys()
        if k[:prefix_len] == prefix
    ]
    return max(keys) if keys else 0


def get_nsequences(state_dict: StateDict, seq_key: str) -> int:
    nsequences: int = get_max_indice(state_dict=state_dict, seq_key=seq_key)
    return nsequences + 1 if nsequences else 0


def get_scale_and_out_nc(x: int, input_channels: int) -> tuple[int, int]:
    """
    Returns a scale and number of output channels such that `scale**2 * out_nc = x`.

    This is commonly used for pixelshuffel layers.
    """
    # Unfortunately, we do not have enough information to determine both the scale and
    # number output channels correctly *in general*. However, we can make some
    # assumptions to make it good enough.
    #
    # What we know:
    # - x = scale * scale * output_channels
    # - output_channels is likely equal to input_channels
    # - output_channels and input_channels is likely 1, 3, or 4
    # - scale is likely 1, 2, 4, or 8

    # just try out a few candidates and see which ones fulfill the requirements
    mod_candidates = (input_channels, 3, 4, 1)
    for c in mod_candidates:
        if x % c == 0:
            v = math.sqrt(x // c)
            if v.is_integer():
                return int(v), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4."
        f" Could not find a pair (scale, out_nc) such that `scale**2 * out_nc = {x}`"
    )



def get_pixelshuffle_params(
    state_dict: StateDict,
    upsample_key: str = "upsample",
    default_nf: int = 64,
) -> tuple[int, int]:
    """
    This will detect the upscale factor and number of features of a pixelshuffle module in the state dict.

    A pixelshuffle module is a sequence of alternating up convolutions and pixelshuffle.
    The class of this module is commonyl called `Upsample`.
    Examples of such modules can be found in most SISR architectures, such as SwinIR, HAT, RGT, and many more.
    """
    upscale = 1
    num_feat = default_nf

    for i in range(0, 10, 2):
        key = f"{upsample_key}.{i}.weight"
        if key not in state_dict:
            break

        tensor: Tensor = state_dict[key]
        # we'll assume that the state dict contains tensors
        shape: tuple[int, ...] = tensor.shape  # type: ignore
        num_feat = shape[1]
        upscale *= math.isqrt(shape[0] // num_feat)

    return upscale, num_feat



def parameters_to_args(
    model: PytorchModel,
    model_class: type
) -> OrderedDict:
    arg_keywords = [kw for kw in inspect.signature(model_class.__init__).parameters.keys()][1:]
    args = OrderedDict()
    for kw in arg_keywords:
        if hasattr(model, kw):
            args[kw] = getattr(model, kw)
    return args


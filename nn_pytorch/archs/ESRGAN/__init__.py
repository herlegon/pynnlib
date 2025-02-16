from collections import OrderedDict
import functools
import math
from pprint import pprint
import re
from pynnlib.utils.p_print import *
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.model import PytorchModel
from ...torch_types import StateDict
from ..torch_to_onnx import to_onnx
from ..helpers import get_max_indice
from .module.RRDB import RRDBNet




def _get_block_count(state: StateDict, state_map: dict) -> int:
    nblocks = []
    state_keys: list[str] = (
        state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"]
        + (r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",)
    )
    for state_key in state_keys:
        for k in state:
            if (m := re.search(state_key, k)):
                nblocks.append(int(m.group(1)))
        if nblocks:
            break
    return max(*nblocks) + 1



def _convert_to_legacy_arch(state_dict: StateDict) -> OrderedDict:
    """Convert a new-arch model state dictionary to an old-arch dictionary."""

    if "params_ema" in state_dict:
        print("[W] params_ema is still in dict_state, should have been removed before")
        state_dict = state_dict["params_ema"]

    if "conv_first.weight" not in state_dict:
        # model is already old arch, this is a loose check, but should be sufficient
        return state_dict

    _esrgan_state_map: dict[str, tuple[str]] = {
        # currently supports old, new, and newer RRDBNet arch models
        # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
        "model.0.weight": ("conv_first.weight",),
        "model.0.bias": ("conv_first.bias",),
        "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
        "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
        r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
            r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
            r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
        ),
    }
    num_blocks: int = _get_block_count(state_dict, _esrgan_state_map)

    # add nb to state keys
    for kind in ("weight", "bias"):
        _esrgan_state_map[f"model.1.sub.{num_blocks}.{kind}"] = _esrgan_state_map[
            f"model.1.sub./NB/.{kind}"
        ]
        del _esrgan_state_map[f"model.1.sub./NB/.{kind}"]

    legacy_state_dict = OrderedDict()
    for old_key, new_keys in _esrgan_state_map.items():
        for new_key in new_keys:
            if r"\1" in old_key:
                for k, v in state_dict.items():
                    sub = re.sub(new_key, old_key, k)
                    if sub != k:
                        legacy_state_dict[sub] = v
            elif new_key in state_dict:
                legacy_state_dict[old_key] = state_dict[new_key]

    # upconv layers
    max_upconv = 0
    for key in state_dict.keys():
        if (match := re.match(re.compile(r"(upconv|conv_up)(\d)\.(weight|bias)"), key)):
            _, key_num, key_type = match.groups()
            legacy_state_dict[f"model.{int(key_num) * 3}.{key_type}"] = state_dict[key]
            max_upconv = max(max_upconv, int(key_num) * 3)

    # final layers
    for key, state_value in state_dict.items():
        if key in ("HRconv.weight", "conv_hr.weight"):
            legacy_state_dict[f"model.{max_upconv + 2}.weight"] = state_value
        elif key in ("HRconv.bias", "conv_hr.bias"):
            legacy_state_dict[f"model.{max_upconv + 2}.bias"] = state_value
        elif key in ("conv_last.weight",):
            legacy_state_dict[f"model.{max_upconv + 4}.weight"] = state_value
        elif key in ("conv_last.bias",):
            legacy_state_dict[f"model.{max_upconv + 4}.bias"] = state_value

    # Sort by first numeric value of each layer
    def compare(item1: str, item2: str):
        return (
            int(item1.split(".", maxsplit=2)[1])
            - int(item2.split(".", maxsplit=2)[1])
        )
    sorted_keys = sorted(
        legacy_state_dict.keys(),
        key=functools.cmp_to_key(compare)
    )

    # Rebuild the state dict in the correct order
    out_dict = OrderedDict((k, legacy_state_dict[k]) for k in sorted_keys)

    return out_dict


def _get_scale(state_dict: StateDict, min_part: int = 6) -> int:
    n = 0
    for part in list(state_dict):
        parts = part.split(".")
        if len(parts) == 3:
            if parts[2] != "weight":
                continue
            if int(parts[1]) > min_part:
                n += 1
    return 2**n



def parse(model: PytorchModel) -> None:
    state_dict: StateDict = model.state_dict
    state_dict = _convert_to_legacy_arch(state_dict)

    first_shape = state_dict["model.0.weight"].shape
    in_nc = first_shape[1]

    max_indice = get_max_indice(state_dict, "model")
    out_nc: int = state_dict[f"model.{max_indice}.bias"].shape[0]

    scale: int = _get_scale(state_dict)

    num_blocks = get_max_indice(state_dict, "model.1.sub")
    num_filters = first_shape[0]

    c2x2: bool = False
    if first_shape[-2] == 2:
        c2x2 = True
        scale = round(math.sqrt(scale / 4))

    # Detect if pixelunshuffle was used (Real-ESRGAN)
    shuffle_factor: int | None = None
    if (
        in_nc in (out_nc * 4, out_nc * 16)
        and out_nc in (in_nc / 4, in_nc / 16)
    ):
        shuffle_factor = int(math.sqrt(in_nc / out_nc))
        in_nc //= shuffle_factor**2
        scale //= shuffle_factor

    # Update model parameters
    model.update(
        state_dict=state_dict,
        arch_name="ESRGAN-2c2" if c2x2 else model.arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=RRDBNet,
        num_filters=num_filters,
        num_blocks=num_blocks,
        c2x2=c2x2,
        shuffle_factor=shuffle_factor,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="BSRGAN+_RealSR",
        detection_keys=(
            "conv_first.weight",
            "RRDB_trunk.0.RDB1.conv1.weight",
            "trunk_conv.weight",
            "conv_last.weight",
        )
    ),
    NnPytorchArchitecture(
        name="ESRGAN+",
        detection_keys=(
            "model.0.weight",
            "model.1.sub.0.RDB1.conv1x1.weight",
        )
    ),
    NnPytorchArchitecture(
        # (legacy)
        # note: addedd a space at the end because
        # detection has not or
        name="ESRGAN ",
        detection_keys=(
            "conv_first.weight",
            "body.0.rdb1.conv1.weight",
            "conv_body.weight",
            "conv_last.weight",
        )
    ),
    NnPytorchArchitecture(
        name="ESRGAN",
        detection_keys=(
            "model.0.weight",
            "model.1.sub.0.RDB1.conv1.0.weight"
        )
    ),
)


for arch in MODEL_ARCHITECTURES:
    arch.parse = parse
    arch.to_onnx = to_onnx
    arch.dtypes = ['fp32', 'fp16']


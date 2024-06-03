import sys
import numpy as np
import torch
from torch import Tensor


# https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
np_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

if sys.platform == "win32":
    # Size of `np.intc` is platform defined.
    # It is returned by functions like `bitwise_not`.
    # On Windows `int` is 32-bit
    # https://docs.microsoft.com/en-us/cpp/cpp/data-type-ranges?view=msvc-160
    np_to_torch_dtype_dict[np.intc] = torch.int

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in np_to_torch_dtype_dict.items()}
torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})

# numpy dtypes like np.float64 are not instances, but rather classes. This leads to rather absurd cases like
# np.float64 != np.dtype("float64") but np.float64 == np.dtype("float64").type.
# Especially when checking against a reference we can't be sure which variant we get, so we simply try both.
def np_to_torch_dtype(np_dtype):
    try:
        return np_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return np_to_torch_dtype_dict[np_dtype.type]



def to_nchw(t: Tensor) -> Tensor:
    shape_size = len(t.shape)
    if shape_size == 3:
        # (H, W, C) -> (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0)

    elif shape_size == 2:
        # (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)

    else:
        raise ValueError("Unsupported input tensor shape")


def to_hwc(t: Tensor) -> Tensor:
    if len(t.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return t.squeeze(0).permute(1, 2, 0)

    else:
        raise ValueError("Unsupported output tensor shape")


def flip_r_b_channels(t: Tensor) -> Tensor:
    if t.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return t.flip(2)

    elif t.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return torch.cat(
            (
                t[:, :, 2],
                t[:, :, 1],
                t[:, :, 0],
                t[:, :, 3]
            ),
            axis=2
        )

    return t


import numpy as np


def to_nchw_np(tensor: np.ndarray) -> np.ndarray:
    shape_size = len(tensor.shape)
    if shape_size == 3:
        # (H, W, C) -> (1, C, H, W)
        return tensor.transpose((2, 0, 1))[np.newaxis, :]

    elif shape_size == 2:
        # (H, W) -> (1, 1, H, W)
        return tensor[np.newaxis, np.newaxis, :]

    else:
        raise ValueError("Unsupported input tensor shape")


def to_hwc_np(tensor: np.ndarray) -> np.ndarray:
    if len(tensor.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return tensor.squeeze(0).transpose(1, 2, 0)

    else:
        raise ValueError("Unsupported output tensor shape")


def flip_r_b_channels_np(tensor: np.ndarray) -> np.ndarray:
    if tensor.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return np.flip(tensor, 2)

    elif tensor.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return np.dstack(
            (
                tensor[:, :, 2],
                tensor[:, :, 1],
                tensor[:, :, 0],
                tensor[:, :, 3]
            )
        )

    return tensor


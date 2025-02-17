import cupy as cp


class MemcpyKind:
    """
    Enumerates different kinds of copy operations.
    """
    HostToHost = 0
    """Copies from host memory to host memory"""
    HostToDevice = 1
    """Copies from host memory to device memory"""
    DeviceToHost = 2
    """Copies from device memory to host memory"""
    DeviceToDevice = 3
    """Copies from device memory to device memory"""
    Default = 4



def to_nchw_cp(tensor: cp.ndarray) -> cp.ndarray:
    shape_size = len(tensor.shape)
    if shape_size == 3:
        # (H, W, C) -> (1, C, H, W)
        return cp.expand_dims(cp.transpose(tensor, (2, 0, 1)), 0)

    elif shape_size == 2:
        # (H, W) -> (1, 1, H, W)
        return tensor[cp.newaxis, cp.newaxis, :]

    else:
        raise ValueError("Unsupported input tensor shape")


def to_hwc_cp(tensor: cp.ndarray) -> cp.ndarray:
    if len(tensor.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return cp.transpose(cp.squeeze(tensor, 0), (1, 2, 0))

    else:
        raise ValueError("Unsupported output tensor shape")


def flip_r_b_channels_cp(tensor: cp.ndarray) -> cp.ndarray:
    # flip returns a copy
    # what about stack?
    # inplace
    # if len(x_gpu_in.shape) != 3:
    #     return x_gpu_in
    if tensor.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return cp.flip(tensor, 2)

    elif tensor.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return cp.stack(
            (
                tensor[:, :, 2],
                tensor[:, :, 1],
                tensor[:, :, 0],
                tensor[:, :, 3]
            ),
            axis=2
        )

    return tensor

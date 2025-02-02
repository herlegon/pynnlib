from ..import_libs import is_cuda_available

if is_cuda_available():
    from .cp_tensor import (
        MemcpyKind,
        flip_r_b_channels,
        to_nchw,
        to_hwc,
    )
else:
    from .torch_tensor import (
        flip_r_b_channels_torch as flip_r_b_channels,
        to_nchw_torch as to_nchw,
        to_hwc_torch as to_hwc,
    )

    class MemcpyKind:
        pass

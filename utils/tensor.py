from ..import_libs import is_cuda_available

if is_cuda_available():
    from .cp_tensor import (
        MemcpyKind,
        flip_r_b_channels_cp,
        to_nchw_cp,
        to_hwc_cp,
    )
else:
    from .torch_tensor import (
        flip_r_b_channels as flip_r_b_channels,
        to_nchw as to_nchw,
        to_hwc as to_hwc,
    )

    class MemcpyKind:
        pass

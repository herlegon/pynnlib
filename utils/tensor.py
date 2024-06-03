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
        np_to_torch_dtype,
        flip_r_b_channels,
        to_nchw,
        to_hwc,
    )

    class MemcpyKind:
        pass

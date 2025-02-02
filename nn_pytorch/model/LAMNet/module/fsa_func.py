import os
import warnings
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp import custom_bwd, custom_fwd
from .cuda_ext import compile_cuda_ext


# Compile CUDA plugin
_inited: bool = False
_fsa_cuda_ext = None

def compile_fsa_ext(verbose: bool = False):
    global _inited
    global _fsa_cuda_ext

    if not _inited:
        _inited = True
        try:
            _fsa_cuda_ext = compile_cuda_ext(
                "fsa_cuda",
                sources=(
                    "fsa/fsa_cuda.cu",
                    "fsa/vision.cpp"
                ),
                extra_include_paths=(
                    os.path.join(os.path.dirname(__file__), "fsa"),
                ),
                verbose=verbose,
                # extra_cuda_cflags=["--use_fast_math", "-O2"],
            )
        except:
            warnings.warn("Failed to build CUDA kernels for bias_act. Falling back to slow reference implementation. Details:\n\n" + traceback.format_exc())
            raise

        return None

    else:
        raise RuntimeError("should be runned onlyce once")

# compile_fsa_ext()


class FSAVFunction(Function):
    @staticmethod
    # @custom_fwd
    def forward(
        ctx,
        input,
        kernel,
        kernel_size,
        kernel_map,
        groups,
        group_channels,
    ):
        ctx.groups = groups
        ctx.group_channels = group_channels
        ctx.kernel_size = kernel_size
        args = [
            input,
            kernel,
            kernel_size,
            kernel_map,
            groups,
            group_channels,
        ]

        output = _fsa_cuda_ext.fsa_vertical_forward(*args)
        ctx.save_for_backward(input, kernel, kernel_map)

        return output

    @staticmethod
    @once_differentiable
    # @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        input, kernel, kernel_map = ctx.saved_tensors
        groups = ctx.groups
        group_channels = ctx.group_channels
        kernel_size = ctx.kernel_size
        args = [
            grad_output,
            input,
            kernel,
            kernel_size,
            kernel_map,
            groups,
            group_channels,
        ]
        grad_input, grad_kernel = _fsa_cuda_ext.fsa_vertical_backward(*args)
        return grad_input, grad_kernel, None, None, None, None, None


class FSAHFunction(Function):
    @staticmethod
    # @custom_fwd
    def forward(
        ctx,
        input,
        kernel,
        kernel_size,
        kernel_map,
        groups,
        group_channels,
    ):
        ctx.save_for_backward(input, kernel, kernel_map)
        ctx.groups = groups
        ctx.group_channels = group_channels
        ctx.kernel_size = kernel_size
        args = [
            input,
            kernel,
            kernel_size,
            kernel_map,
            groups,
            group_channels,
        ]

        output = _fsa_cuda_ext.fsa_horizontal_forward(*args)

        return output

    @staticmethod
    @once_differentiable
    # @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        input, kernel, kernel_map = ctx.saved_tensors
        groups = ctx.groups
        group_channels = ctx.group_channels
        kernel_size = ctx.kernel_size
        args = [
            grad_output,
            input,
            kernel,
            kernel_size,
            kernel_map,
            groups,
            group_channels,
        ]
        grad_input, grad_kernel = _fsa_cuda_ext.fsa_horizontal_backward(*args)
        return grad_input, grad_kernel, None, None, None, None, None


def fsa_spatial(input, kernel, kernel_size, kernel_map, groups, group_channels, direction):
    if direction == "vertical":
        return FSAVFunction.apply(input, kernel, kernel_size, kernel_map, groups, group_channels)
    elif direction == "horizontal":
        return FSAHFunction.apply(input, kernel, kernel_size, kernel_map, groups, group_channels)
    else:
        raise NotImplementedError("Direction not implemented")

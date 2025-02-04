import os
from pathlib import Path
from typing import Any, Type

from pynnlib.session import set_cuda_device
import tensorrt as trt
from tensorrt import DataType as TrtDType

from pynnlib.architecture import detect_model_arch
from pynnlib.nn_types import NnModelDtype
from pynnlib.utils.p_print import *
from pynnlib.model import ShapeStrategy, TrtModel
from pynnlib.logger import nnlogger
from ..trt_types import TrtEngine
from ..inference.session import (
    TensorRtSession,
    TRT_LOGGER,
)



def get_model_arch(
    nn_model_path: str | Path,
    nn_arch_database: dict[str, dict],
    device: str = 'cuda'
) -> tuple[str, str | Path | None]:
    arch_name = detect_model_arch(nn_model_path, nn_arch_database)
    return arch_name, nn_model_path



def is_model_generic(model: Path | str | TrtEngine) -> bool:
    """Always onsider this model as a generic one
    """
    if isinstance(model, Path | str) and os.path.exists(model):
        return True
    elif isinstance(model, TrtEngine):
        return True
    return False



def get_shapes_dtype(engine) -> dict[str, tuple[str | int | Any]]:
    """Returns a dict of shape and dtype as a tuple of [name, (B, C, H, W), dtype]
        for each input and output tensors
    """
    io_tensors: dict[str, list] = {
        'inputs': [],
        'outputs': [],
    }
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        dtype = engine.get_tensor_dtype(tensor_name)

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            io_tensors['inputs'].append((tensor_name, shape, dtype))
        elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            io_tensors['outputs'].append((tensor_name, shape, dtype))

    return io_tensors



def get_shape_strategy(engine, tensor_name: str) -> ShapeStrategy:
    """Extract the shape strategy of the specified tensor
    Returns only the 1st profile found
    """
    shape_strategy = ShapeStrategy()
    min_shapes, opt_shapes, max_shapes = engine.get_tensor_profile_shape(tensor_name, 0)
    shape_strategy.min_size = tuple(reversed(min_shapes[2:]))
    shape_strategy.opt_size = tuple(reversed(opt_shapes[2:]))
    shape_strategy.max_size = tuple(reversed(max_shapes[2:]))
    return shape_strategy



def parse_engine(model: TrtModel) -> None:
    set_cuda_device(model.device)
    engine = None
    if os.path.exists(model.filepath):
        trt_runtime = trt.Runtime(TRT_LOGGER)
        with open(model.filepath, 'rb') as f:
            engine_data = f.read()
        try:
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
        except:
            print("[E] Not a valid engine")

    elif model.engine is not None:
        engine = model.engine

    if engine is None:
        raise ValueError("[E] Not a compatible engine")

    tensor_shapes_dtype = get_shapes_dtype(engine)
    if len(tensor_shapes_dtype['inputs']) != 1:
        raise NotImplementedError(f"TensorRT: unsupported nb of inputs ({len(tensor_shapes_dtype['inputs'])})")
    if len(tensor_shapes_dtype['inputs']) != 1:
        raise NotImplementedError(f"TensorRT: unsupported nb of outputs ({len(tensor_shapes_dtype['outputs'])})")
    input_name, shape, in_dtype = tensor_shapes_dtype['inputs'][0]
    in_b, in_nc, in_h, in_w = shape
    _, shape, out_dtype = tensor_shapes_dtype['outputs'][0]
    _, out_nc, out_h, out_w = shape
    if in_dtype != out_dtype:
        raise NotImplementedError("TensorRT: IO, dtypes are not the same")

    dtypes: set[NnModelDtype] = set()
    if in_dtype == TrtDType.FLOAT :
        dtypes.add('fp32')
    elif in_dtype == TrtDType.HALF:
        dtypes.add('fp16')
    elif in_dtype == TrtDType.BF16:
        dtypes.add('bf16')
    else:
        raise ValueError(f"TensorRT: datatype {in_dtype} is not supported")

    # TODO: get shape strategy for each profile?
    shape_strategy = get_shape_strategy(engine, input_name)

    if any(x == -1 for x in (in_w, in_h, out_w, out_h)):
        # Dynamic shapes
        # https://www.programcreek.com/python/?code=tensorboy%2Fcenterpose%2Fcenterpose-master%2Fdemo%2Ftensorrt_model.py
        # https://forums.developer.nvidia.com/t/how-to-inference-with-2-different-shape-outputs/165468
        if all(x != 0 for x in shape_strategy.min_size):
            input_shape: tuple[tuple] = (in_b, in_nc, *reversed(shape_strategy.min_size))
        else:
            input_shape: tuple[tuple] = (in_b, in_nc, *reversed(shape_strategy.opt_size))

        output_shapes: tuple[tuple[int, int, int, int]] = []
        # By default, use the first profile (context.active_optimization_profile = 0)
        with engine.create_execution_context() as context:
            for tensor in tensor_shapes_dtype['inputs']:
                context.set_input_shape(tensor[0], input_shape)
            for tensor in tensor_shapes_dtype['outputs']:
                output_shapes.append(context.get_tensor_shape(tensor[0]))

        if len(output_shapes) > 1:
            raise NotImplementedError("TensorRT: unsupported multiple outputs")
        scale_h, scale_w = [o // i for o, i in zip(output_shapes[0][2:], input_shape[2:])]
        if scale_h != scale_w:
            raise NotImplementedError(f"TensorRT: \'width\' scale ({scale_w}) differs from \'height\' scale ({scale_h})")
        scale = scale_w
    else:
        # raise NotImplementedError("TensorRT: static shapes not yet supported")
        scale_h, scale_w = out_h//in_h, out_w//in_w,
        if scale_h != scale_w:
            raise NotImplementedError(f"TensorRT: \'width\' scale ({scale_w}) differs from \'height\' scale ({scale_h})")
        scale = scale_w

    model.update(
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        dtypes=dtypes,
        engine=engine,
        shape_strategy=shape_strategy,
    )


def create_session(model: TrtModel) -> Type[TensorRtSession]:
    return model.framework.Session(model)


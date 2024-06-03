from functools import partial
import onnx
from typing import Literal
from pynnlib.utils.p_print import *
from pynnlib.logger import nnlogger
from pynnlib.architecture import NnOnnxArchitecture
from pynnlib.model import OnnxModel
from ..inference.session import OnnxSession
from .onnx_to_tensorrt import to_tensorrt
from .parser import parse


def is_model_bgd_removal(model: onnx.ModelProto, item: Literal['node', 'output'], keys: tuple[str]) -> bool:
    if len(model.graph.input) != 1:
        # raise ValueError("ONNX model: input length shall be equal to 1")
        return False

    if item == 'node':
        parsed = tuple([gn for gn in model.graph.node[-2].input])
    elif item == 'output':
        parsed = tuple([go.name for go in model.graph.output])
    return True if parsed == keys else False


def is_model_rife(model: onnx.ModelProto) -> bool:
    # Detect ONNX rife models
    # highly experimental because not enough unique keys/nodes
    graph = model.graph
    if len(graph.input) != 1:
        # raise ValueError("ONNX model: input length shall be equal to 1")
        return False

    shape = graph.input[0].type.tensor_type.shape
    input_shape = [dim.dim_value for dim in shape.dim]

    shape = graph.output[0].type.tensor_type.shape
    output_shape = [dim.dim_value for dim in shape.dim]

    try:
        if (graph.node[1].input[0] == graph.input[0].name
            and graph.node[1].op_type == 'Split'
            and input_shape[1] == 11

            and graph.node[-1].output[0] == graph.output[0].name
            and output_shape[1] == 3
            and graph.node[-1].op_type == 'Add'
        ):
            nnlogger.debug(yellow("rife"))
            return True
    except:
        pass
    return False


def is_model_generic(model: onnx.ModelProto) -> bool:
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        nnlogger.debug("[E] Model with more than single i/o is not supported")
        return False

    return True


def create_session(model: OnnxModel) -> OnnxSession:
    nnlogger.debug(yellow("load onnx model"))
    return OnnxSession(model)



MODEL_ARCHITECTURES: tuple[NnOnnxArchitecture] = (
    NnOnnxArchitecture(
        name='Rife',
        category='Frame interpolation',
        detect=is_model_rife,
    ),
    NnOnnxArchitecture(
        name='Generic',
        detect=is_model_generic,
        to_tensorrt=to_tensorrt,
    )
)

# Append common variables for all architectures
for arch in MODEL_ARCHITECTURES:
    arch.update(dict(
        type=arch.name,
        parse=partial(parse, scale=arch.scale),
        create_session=create_session,
        dtypes = ['fp32', 'fp16']
    ))

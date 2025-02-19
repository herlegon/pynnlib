from copy import deepcopy
import math
from warnings import warn
import onnx
import onnxruntime as ort

from pynnlib.logger import nnlogger
from pynnlib.nn_types import NnModelDtype
nnlogger.debug(f"[I] ONNX runtime package loaded (version {ort.__version__})")

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph
from pathlib import Path
from typing import Literal

from pynnlib.architecture import (
    NnOnnxArchitecture,
    detect_model_arch
)
from pynnlib.model import OnnxModel
from .ms_utils import (
    make_input_shape_fixed,
    fix_output_shapes
)
from pynnlib.utils.p_print import *



OnnxShapeOrder = Literal['NCHW', 'NHWC']


def get_opset_version(model_proto: onnx.ModelProto) -> int:
    return model_proto.opset_import[0].version



def get_input_datatype(model: onnx.ModelProto):
    """Returns the datatype of the first input of the model.
    It may return an erroneous datatype when there are multiple inputs"""
    input = model.graph.input[0]
    dim_0 = str(input.type.tensor_type.elem_type).split()[0]
    try:
        return onnx.helper.tensor_dtype_to_string(int(dim_0))
    except:
        pass
    return None

    # nnlogger.debug("onnx.TensorProto.DataType")
    # nnlogger.debug(onnx.TensorProto.DataType)
    # onnx.TensorProto.DataType.Float16
    # if tensor_dtype is not None:
    #     np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_dtype)
    # return np_dtype



def guess_shape(
    tensor_shape: tuple[int, int, int, int]
) -> tuple[OnnxShapeOrder, int, int, int]:
    """Return tensor format followed by hwc"""
    img_shape = tensor_shape[1:]
    simplified_shape = [x for x in img_shape if x > 0]
    min_value: int = (
        min(simplified_shape)
        if simplified_shape
        else max(min(img_shape), 0)
    )

    dim1, dim2, dim3 = img_shape
    if dim1 == min_value:
        return 'NCHW', dim2, dim3, dim1

    elif dim3 == min_value:
        return 'NHWC', dim1, dim2, dim3

    return 'NCHW', dim2, dim3, dim1




def get_io_shapes_from_graph(model: onnx.ModelProto) -> tuple[tuple[int, str]]:
    # Single input/output currently supported
    graph: onnx.GraphProto = model.graph
    in_field, out_field = graph.input[0], graph.output[0]
    in_shape = [dim.dim_value for dim in in_field.type.tensor_type.shape.dim]
    out_shape = [dim.dim_value for dim in out_field.type.tensor_type.shape.dim]
    # nnlogger.debug(f"[V] ONNX: get io shapes from graph, in={in_shape}, out={out_shape}")
    return in_shape, out_shape



def get_shapes_from_shape_inference(
    model_proto: onnx.ModelProto,
    shape_order: OnnxShapeOrder,
    dummy_size: int = 32,
    in_nc: int = 3,
) -> tuple[tuple[int], int]:
    # Modify the graph with a fixed input shape
    # For single input  model
    if in_nc not in (1, 3):
        in_nc = 3

    # Create a dummy tensor dim
    if shape_order == 'NCHW':
        dummy: tuple[int] = (1, in_nc, dummy_size, dummy_size)
    else:
        dummy: tuple[int] = (1, dummy_size, dummy_size, in_nc)
    make_input_shape_fixed(model_proto, dummy)

    # Shape inference
    fix_output_shapes(model_proto)
    in_shape, out_shape = get_io_shapes_from_graph(model_proto)
    return in_shape, out_shape



def get_shapes_from_session(
    model_proto: onnx.ModelProto
) -> tuple[tuple[int, int, int, int]]:
    # Works for a single input only
    input_shape: tuple[int, int, int, int] = [0] * 4
    output_shape: tuple[int, int, int, int] = [0] * 4
    nnlogger.debug(purple("[W] ONNX: get shapes from session"))
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            model_proto.SerializeToString(),
            sess_options=options
        )
    except InvalidGraph as e:
        nnlogger.debug(red("[E] this model is invalid (graph)"))
        return
    except Exception as e:
        if "ShapeInferenceError" in f"{e}":
            # Do not raise an error as it not currently catched by caller #TODO
            nnlogger.debug(red("[E] Invalid model due to invalid shape dimensions"))
            return input_shape, output_shape
        # Do not raise an error as it not currently catched by caller #TODO
        nnlogger.debug(red(f"[E] ONNX session failed: {type(e)}"))
        return input_shape, output_shape

    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    return input_shape, output_shape



def get_scale_from_shape_inference(
    model_proto: onnx.ModelProto,
    in_shape_order: OnnxShapeOrder,
    in_nc: int = 3,
) -> tuple[int, int, int]:
    scale_h, scale_w, out_nc = 0, 0, 0
    try:
        onnx.checker.check_model(model_proto)
    except Exception as e:
        nnlogger.debug(red(f"[W] Not a valid Onnx model: {type(e)}"))

    model_proto_tmp: onnx.ModelProto = deepcopy(model_proto)

    # TODO: change this hardcoded size by the min size?
    try:
        in_shape, out_shape = get_shapes_from_shape_inference(
            model_proto=model_proto_tmp,
            shape_order=in_shape_order,
            dummy_size=64,
            in_nc=in_nc
        )
        nnlogger.debug(f"[V] shape inference returned: in={in_shape}, out={out_shape}")
    except:
        warn(f"[V] failed to detect scale, default to 2")
        return 2, 2, 3


    # If shape inference "failed", create an onnx session
    img_shape = guess_shape(out_shape)[1:]
    if math.prod(img_shape) == 0:
        in_shape, out_shape = get_shapes_from_session(model_proto_tmp)
        nnlogger.debug(f"[V] shapes from session: in={in_shape}, out={out_shape}")

    # Now, shapes should be valid
    in_shape_order, in_h, in_w, _ = guess_shape(in_shape)
    _, out_h, out_w, out_nc = guess_shape(out_shape)
    try:
        if not any(x < 0 for x in [out_h, in_h, out_w, in_w]):
            scale_h, scale_w = out_h // in_h, out_w // in_w
    except Exception as e:
        # Not a valid model ?
        nnlogger.debug(red("[E] onnx: something went wrong while calculating scales"))
        nnlogger.debug(f"[E]    [oh, ih, ow, iw] = [{out_h}, {in_h}, {out_w}, {in_w}]")

    return scale_h, scale_w, out_nc



def parse(
    model: OnnxModel,
    scale: int | None = None
) -> None:
    model_proto: onnx.ModelProto = model.model_proto

    in_shape, out_shape = get_io_shapes_from_graph(model_proto)
    nnlogger.debug(f"[V] shapes: in={in_shape}, out={out_shape}")
    if len(out_shape) != 4 or len(in_shape) != 4:
        raise ValueError("ONNX model: shape must contains batch, channels, width, height")

    in_shape_order, in_h, in_w, in_nc = guess_shape(in_shape)
    out_shape_order, out_h, out_w, out_nc = guess_shape(out_shape)

    # Calculate scales if possible
    nnlogger.debug(f"[V] shape order: in={in_shape_order}, out={out_shape_order}")
    is_scale_valid: bool = False
    if scale is not None and scale > 0:
        is_scale_valid = True
    else:
        try:
            if not any(x < 0 for x in [out_h, in_h, out_w, in_w]):
                scale_h, scale_w = out_h // in_h, out_w // in_w
                # Note: is downscaling valid? -> yes
                if scale_h == scale_w and scale_w > 0:
                    is_scale_valid = True
                    scale = scale_w
        except:
            pass

    if not is_scale_valid:
        nnlogger.debug("[V] scale is not valid, perform a shape inference")
        scale_h, scale_w, out_nc = get_scale_from_shape_inference(
            model_proto, in_shape_order, in_nc
        )

    # Consolidate scale
    scale = None
    if scale_w is not None and scale_w == scale_h:
        scale = scale_w
    if scale is None or scale == 0:
        raise ValueError(red("[E] onnx: unsupported scales, W and H scales must be identical and > 0"))
        # raise ValueError("ONNX model: scale shall be different from 0")
        # Do not raise an error as it is not currently catched by caller #TODO

    # Channels
    if isinstance(out_nc, str):
        out_nc = 3

    # Required size
    # exact_size = None
    # if req_width is not None:
    #     exact_size = req_width, req_height or req_width
    # elif req_height is not None:
    #     exact_size = req_width or req_height, req_height

    # TODO, one day: get datatype for all inputs
    onnx_dtype_to_str: dict[str, NnModelDtype] = {
        'TensorProto.FLOAT': 'fp32',
        'TensorProto.FLOAT16': 'fp16',
        'TensorProto.BFLOAT16': 'bf16',
    }
    supported_dtypes = set(
        [onnx_dtype_to_str[get_input_datatype(model_proto)]]
    )

    model.update(
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        opset=get_opset_version(model_proto),
        dtypes=supported_dtypes,

        # exact_size=exact_size,
        in_shape_order=in_shape_order,
    )



    # nnlogger.debug(purple(f"opset: {model.opset} [{f', '.join(model.dtypes)}]"))



def get_model_arch(
    model_object: str | Path | onnx.ModelProto,
    architectures: dict[str, NnOnnxArchitecture],
    device: str = 'cpu'
) -> tuple[NnOnnxArchitecture|None, onnx.ModelProto]:

    onnx_model = (
        onnx.load_model(model_object)
        if isinstance(model_object, str | Path)
        else model_object
    )
    arch = detect_model_arch(onnx_model, architectures)
    return arch, onnx_model



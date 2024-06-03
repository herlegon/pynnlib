import numpy as np
from pynnlib.session import set_cuda_device
from pynnlib.utils.p_print import *
import tensorrt as trt
from pynnlib.model import OnnxModel
from ..inference.session import TRT_LOGGER
from ..trt_types import ShapeStrategy, TrtEngine



# https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/builder.py#L405

def _onnx_to_trt_engine(
    model: OnnxModel,
    device: str,
    fp16: bool,
    bf16: bool,
    shape_strategy: ShapeStrategy,
) -> TrtEngine:
    """
    Convert an ONNX model to a serialized TensorRT engine.

    Restrictions:
        support only a single input tensor

    """
    print("[V] Start converting to TRT")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    set_cuda_device(device)
    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(flags=network_flags) as network,
        trt.OnnxParser(network, TRT_LOGGER) as onnx_parser
    ):
        runtime = trt.Runtime(TRT_LOGGER)

        # Parse the serializedonnx model
        if model.model_proto is not None:
            # Serialize the onnx model. Don't use filepath as it may
            # not an optimized onnx model
            serialized_onnx_model = model.model_proto.SerializeToString()
            success = onnx_parser.parse(serialized_onnx_model)
        if not success:
            for idx in range(onnx_parser.num_errors):
                print(f"Error: {onnx_parser.get_error(idx)}")
            raise ValueError("Failed to parse the onnx model")

        # Input tensors
        input_tensor = None
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            break
        if input_tensor is None:
            raise ValueError("Missing input tensor in model")
        input_name = input_tensor.name
        is_fp16 = True if trt.nptype(input_tensor.dtype) == np.float16 else fp16
        print(f"is_fp16: {is_fp16}, to_fp16: {fp16}")
        # builder.max_batch_size = 1

        # Create a build configuration specifying how TensorRT should optimize the model
        builder_config = builder.create_builder_config()

        # builder_config.set_flag(trt.BuilderFlag.REFIT)
        # 1GB
        # builder_config.max_workspace_size = 1 << 30
        # builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # builder_config.avg_timing_iterations = 10

        # tactic_source = builder_config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
        # builder_config.set_tactic_sources(tactic_source)

        # builder_config.set_preview_feature(
        #     trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, False)

        # Set config:
        # config.
        #   num_optimization_profiles
        #   profiling_verbosity
        #   engine_capability
        #   builder_optimization_level
        #   hardware_compatibility_level
        #   flags
        # config.flag

        # optimization level: default=3
        # builder.builder_optimization_level = 5

        if is_fp16 or fp16:
            if builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
            else:
                raise RuntimeError("Error: fp16 is requested but this platform does not support it")

        # TODO create a list of profiles for each input
        profile = builder.create_optimization_profile()
        batch_opt = 1
        if not shape_strategy.static:
            profile.set_shape(
                input=input_name,
                min=(batch_opt, model.in_nc, *reversed(shape_strategy.min_size)),
                opt=(batch_opt, model.in_nc, *reversed(shape_strategy.opt_size)),
                max=(batch_opt, model.in_nc, *reversed(shape_strategy.max_size)),
            )
        builder_config.add_optimization_profile(profile)

        # profile_batch_min = builder.create_optimization_profile()
        # if not shape_strategy.static:
        #     profile.set_shape(
        #         input=input_name,
        #         min=(batch_min, model.in_nc, *reversed(shape_strategy.min_shape)),
        #         opt=(batch_min, model.in_nc, *reversed(shape_strategy.opt_shape)),
        #         max=(batch_min, model.in_nc, *reversed(shape_strategy.max_shape)),
        #     )
        # builder_config.add_optimization_profile(profile_batch_min)

        print("[I] Building a TensortRT engine; this may take a while...")
        # serialized_engine = builder.build_serialized_network(network, config)
        engine_bytes = builder.build_serialized_network(network, builder_config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create Tensor RT engine")
        trt_engine = runtime.deserialize_cuda_engine(engine_bytes)

        # buffer = trt_engine.serialize()
        # with open("A:\\ml_models\\trt_engine_v10.engine", 'wb') as trt_engine_file:
        #     trt_engine_file.write(buffer)

    return trt_engine



def onnx_to_trt_engine(
    model: OnnxModel,
    device: str,
    fp16: bool,
    bf16: bool,
    shape_strategy: ShapeStrategy,
):
    engine = _onnx_to_trt_engine(
        model,
        device,
        fp16,
        bf16,
        shape_strategy=shape_strategy
    )
    return engine


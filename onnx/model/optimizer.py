import onnx
import onnxoptimizer


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        # https://github.com/onnx/optimizer/blob/master/examples/onnx_optimizer_exec.cpp
        onnx.checker.check_model(model=model)
        model = onnxoptimizer.optimize(
            model=model,
            passes=onnxoptimizer.get_fuse_and_elimination_passes()
        )
        onnx.checker.check_model(model=model)
    except:
        print("ONNX optimizer: failed, TODO: investigate why")
        pass

    return model

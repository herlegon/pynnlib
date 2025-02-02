import numpy as np
from pprint import pprint
from pynnlib import is_tensorrt_available
from utils.p_print import yellow
if is_tensorrt_available():
    import tensorrt as trt

import torch
from pynnlib.logger import nnlogger
from pynnlib.model import TrtModel
from pynnlib.session import GenericSession, set_cuda_device
from pynnlib.utils.torch_tensor import (
    flip_r_b_channels_torch,
    to_nchw_torch,
    to_hwc_torch,
)


class TrtLogger(trt.ILogger):
    SEVERITY_LETTER_MAPPING: dict = {
        trt.ILogger.INTERNAL_ERROR: "[!]",
        trt.ILogger.ERROR: "[E]",
        trt.ILogger.WARNING: "[W]",
        trt.ILogger.INFO: "[I]",
        trt.ILogger.VERBOSE: "[V]",
    }

    def __init__(self, min_severity):
        trt.ILogger.__init__(self)
        self._min_severity = min_severity

    @property
    def severity(self):
        return self._min_severity

    @severity.setter
    def severity(self, value) -> None:
        self._min_severity = value

    def log(self, severity: trt.ILogger.Severity, msg: str):
        if severity <= self.severity:
            print(f"{self.SEVERITY_LETTER_MAPPING[severity]} [TRT] {msg}")


TRT_LOGGER = TrtLogger(trt.ILogger.INFO)

def get_trt_logger():
    return TRT_LOGGER


class TensorRtSession(GenericSession):
    """An example of session used to perform the inference using a TensorRT engine.
    There is a reason why the initialization is not done in the __init__: mt/mp
    """

    def __init__(
        self,
        model: TrtModel
    ):
        super().__init__()
        self.model: TrtModel = model
        self._infer_stream = None

        # Use the best datatype
        self.fp16 = bool('fp16' in self.model.dtypes)
        self.in_tensor_dtype: torch.dtype = (
            torch.float16 if 'fp16' in self.model.dtypes else torch.float32
        )
        self.out_tensor_dtype: torch.dtype = self.in_tensor_dtype
        self._in_tensor_name: str = 'input'

    @property
    def in_tensor_name(self) -> str:
        return self._in_tensor_name

    @in_tensor_name.setter
    def in_tensor_name(self, name: str) -> None:
        self._in_tensor_name = name

    def initialize(self,
        device: str = 'cuda:0',
        fp16: bool = True,
        **kwargs,
    ):
        # Device and dtype
        self.device = device
        self.fp16 = fp16 and ('fp16' in self.model.dtypes)
        self.in_tensor_dtype = torch.float16 if self.fp16 else torch.float32
        self.out_tensor_dtype = self.in_tensor_dtype

        nnlogger.debug(f"[I] Use {device} to load the tensorRT Engine")
        set_cuda_device(device)

        # Deserialize and create the context
        model_path = self.model.filepath
        print(yellow("TensorRtSession::initialize"))
        trt_runtime = trt.Runtime(get_trt_logger())
        with open(model_path, 'rb') as f:
            serialized_engine = f.read()

        infer_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(infer_stream):
        # with self._infer_stream:
            if self.model.engine is None:
                self.engine = trt_runtime.deserialize_cuda_engine(serialized_engine)
            else:
                self.engine = self.model.engine
            # Create a context (without reusing device memory)
            self.context = self.engine.create_execution_context()

        self.warmup(3)


    def warmup(self, count: int = 1):
        shape = [*reversed(self.model.shape_strategy.opt_size), self.model.in_nc]
        tensor_shape = [
            1,
            self.model.in_nc,
            *reversed(self.model.shape_strategy.opt_size)
        ]
        context, engine = self.context, self.engine
        for idx in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(idx)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor_name, tensor_shape)
        nnlogger.debug(f"[V] warmup ({count}x) with a random img ({shape})")
        img = np.random.random(shape).astype(np.float32)
        for _ in range(count):
            self.process(img)


    def process(self, in_img: np.ndarray) -> np.ndarray | None:
        """This is the worst optimized inference function to perform inference
        of TensorRT engines.
        """
        if in_img.dtype != np.float32:
            raise ValueError("np.float32 img only")

        context, engine = self.context, self.engine
        device = self.device
        in_h, in_w, c = in_img.shape
        out_shape = (
            in_h * self.model.scale,
            in_w * self.model.scale,
            c
        )
        in_tensor_shape = (1, c, *in_img.shape[:2])
        out_tensor_shape = (1, c, *out_shape[:2])
        tensor_dtype = torch.float16 if self.fp16 else torch.float32

        infer_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(infer_stream):
            in_tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(in_img))
            in_tensor = in_tensor.to(device, dtype=torch.float32)
            in_tensor = in_tensor.half() if self.fp16 else in_tensor.float()
            in_tensor = flip_r_b_channels(in_tensor)
            in_tensor = to_nchw(in_tensor)
            in_tensor_shape = in_tensor.shape
            in_tensor = in_tensor.ravel()
            out_tensor: torch.Tensor = torch.empty(
                out_tensor_shape,
                dtype=tensor_dtype,
                device=device
            )

            bindings = [in_tensor.data_ptr(), out_tensor.data_ptr()]
            for i in range(engine.num_io_tensors):
                context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(tensor_name, in_tensor_shape)

            context.execute_async_v3(stream_handle=infer_stream.cuda_stream)

            out_tensor = torch.clamp_(out_tensor, 0, 1)
            out_tensor = to_hwc(out_tensor)
            out_tensor = flip_r_b_channels(out_tensor)
            out_tensor = out_tensor.float()
            infer_stream.synchronize()
            out_img: np.ndarray = out_tensor.detach().cpu().numpy()
        return out_img



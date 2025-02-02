from __future__ import annotations
from pprint import pprint
import re
import onnxruntime as ort
import numpy as np
import torch

from pynnlib.import_libs import is_cuda_available
from pynnlib.model import OnnxModel
from pynnlib.utils.torch_tensor import (
    flip_r_b_channels_torch,
    to_nchw_torch,
    to_hwc_torch,
)


class OnnxSession:
    """An example of session used to perform the inference using an Onnx model.
    """

    def __init__(self, model: OnnxModel):
        self.model: OnnxModel = model
        self.execution_providers: list[str | tuple[str, int]] = [
            "CPUExecutionProvider"
        ]
        self.device: str = 'cpu'
        self.cuda_device_id: int = 0


    def initialize(
        self,
        device: str = 'cpu',
        fp16: bool = False
    ):
        self.execution_providers = ["CPUExecutionProvider"]
        self.device = 'cpu'

        self.cuda_device_id: int = 0
        if 'cuda' in device:
            if (
                is_cuda_available()
                and 'CUDAExecutionProvider' in ort.get_available_providers()
            ):
                if (match := re.match(re.compile(r"^cuda[:]?(\d)?$"), device)):
                    self.cuda_device_id = (
                        int(match.group(1))
                        if match.group(1) is not None
                        else 0
                    )
                    self.execution_providers.insert(
                        0, ('CUDAExecutionProvider', {"device_id": self.cuda_device_id})
                    )
                    self.device = 'cuda'
            else:
                raise ValueError("Unsupported hardware: cuda")

        elif device == 'dml':
            if 'DmlExecutionProvider' in ort.get_available_providers():
                self.execution_providers.insert(
                    0, ('DmlExecutionProvider', {"device_id": self.cuda_device_id})
                )
            else:
                raise ValueError("Unsupported hardware: DirectML")

        model_proto = self.model.model_proto
        # TODO: optimize model?
        # seems not faster for:
        #   - realPKLSR
        # model_proto = optimize_model(model_proto)
        byte_model: bytes = model_proto.SerializeToString()

        # https://onnxruntime.ai/docs/performance/tune-performance/threading.html
        session_options = ort.SessionOptions()
        # session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        try:
            self.session = ort.InferenceSession(
                byte_model,
                sess_options=session_options,
                providers=self.execution_providers
            )
        except:
            raise RuntimeError("Cannot create an Onnx session")

        self.fp16: bool = fp16 and 'fp16' in self.model.dtypes
        if self.fp16 and 'fp16' not in self.model.dtypes:
            raise ValueError("Half datatype (fp16) is not supported by this model")
        if 'fp32' not in self.model.dtypes and device == 'cpu':
            raise ValueError(f"The execution provider ({device}) does not support the datatype of this model (fp16)")

        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name


    def run_np(self, in_img: np.ndarray) -> np.ndarray:
        # Unused: first PoC, use np for image to tensor
        # keep for historical reason
        if in_img.dtype != np.float32:
            raise NotImplementedError("Only float32 input image is supported")

        if self.fp16:
            in_img = in_img.astype(np.float16, copy=False)
        in_img = flip_r_b_channels_torch(in_img)
        in_img = to_nchw_torch(in_img)
        in_img = np.ascontiguousarray(in_img)

        out_img: np.ndarray = self.session.run(
            [self.output_name],
            {self.input_name: in_img}
        )[0]

        out_img = to_hwc_torch(out_img)
        out_img = flip_r_b_channels_torch(out_img)
        out_img = out_img.clip(0, 1., out=out_img)
        if self.fp16:
            out_img = out_img.astype(np.float32)

        return np.ascontiguousarray(out_img)


    def process(self, in_img: np.ndarray) -> np.ndarray:
        if in_img.dtype != np.float32:
            raise ValueError("np.float32 img only")

        in_h, in_w, c = in_img.shape
        out_shape = (
            in_h * self.model.scale,
            in_w * self.model.scale,
            c
        )
        in_tensor_shape = (1, c, *in_img.shape[:2])
        out_tensor_shape = (1, c, *out_shape[:2])
        tensor_dtype = torch.float16 if self.fp16 else torch.float32
        tensor_np_dtype = np.float16 if self.fp16 else np.float32
        session: ort.InferenceSession = self.session
        tensor_device = self.device

        in_tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(tensor_device, dtype=torch.float32)
        in_tensor = in_tensor.half() if self.fp16 else in_tensor.float()
        in_tensor = flip_r_b_channels_torch(in_tensor)
        in_tensor = to_nchw_torch(in_tensor)
        in_tensor = in_tensor.contiguous()

        out_tensor: torch.Tensor = torch.empty(
            out_tensor_shape,
            dtype=tensor_dtype,
            device=tensor_device
        ).contiguous()

        binding = session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type=tensor_device,
            device_id=self.cuda_device_id,
            element_type=tensor_np_dtype,
            shape=tuple(in_tensor_shape),
            buffer_ptr=in_tensor.data_ptr(),
        )
        binding.bind_output(
            name=self.output_name,
            device_type=tensor_device,
            device_id=self.cuda_device_id,
            element_type=tensor_np_dtype,
            shape=tuple(out_tensor_shape),
            buffer_ptr=out_tensor.data_ptr(),
        )

        session.run_with_iobinding(binding)

        out_tensor = torch.clamp_(out_tensor, 0, 1)
        out_tensor = to_hwc_torch(out_tensor)
        out_tensor = flip_r_b_channels_torch(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return np.ascontiguousarray(out_img)


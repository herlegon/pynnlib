from __future__ import annotations
import json
from warnings import warn

from .import_libs import is_tensorrt_available
from .logger import nnlogger
from datetime import datetime
import onnx
nnlogger.debug(f"[I] ONNX package loaded (version {onnx.__version__})")
import os
from pathlib import Path
from pprint import pprint
import re
import time

try:
    from .nn_tensor_rt.trt_types import ShapeStrategy
    from .nn_tensor_rt.archs.save import generate_tensorrt_basename
except:
    # nnlogger.debug("[W] TensorRT is not supported: model cannot be converted")
    def generate_tensorrt_basename(*args) -> str:
        raise RuntimeError("TensorRT is not supported")

import torch
from .utils import (
    get_extension,
    os_path_basename,
)
from .utils.p_print import *

from .architecture import (
    NnArchitecture,
    NnTensorrtArchitecture
)
from .framework import (
    NnFramework,
    import_frameworks,
    extensions_to_framework,
)
from .model import (
    NnModel,
    OnnxModel,
    PytorchModel,
    TrtModel,
)
from .nn_types import (
    Idtype,
    NnModelObject,
    NnFrameworkType
)
from .nn_pytorch.archs.unpickler import RestrictedUnpickle
from .session import NnModelSession



class NnLib:

    def __init__(self) -> None:
        self.frameworks: dict[NnFrameworkType, NnFramework] = import_frameworks()
        nnlogger.debug(
            f"[I] Available frameworks: {', '.join(list([fwk.type.value for fwk in self.frameworks.values()]))}"
        )


    def get_framework_from_extension(self, nn_model_path: str | Path) -> NnFramework:
        extension = get_extension(nn_model_path)
        try:
            return self.frameworks[extensions_to_framework[extension]]
        except KeyError:
            nnlogger.debug(f"[E] No framwework found for model {nn_model_path}, unrecognized extension")
        except:
            raise ValueError(f"No framwework found for model {nn_model_path}")

        return None


    def open(
        self,
        model_path: str | Path,
        device: str = 'cpu',
    ) -> NnModel | None:
        """Open and parse a model and returns its parameters"""
        if not os.path.exists(model_path):
            warn(red(f"[E] {model_path} does not exist"))
            return None

        fwk = self.get_framework_from_extension(model_path)
        if fwk is None:
            warn(f"[E] No framework found for model {model_path}")
            return None

        model_arch, model_obj = fwk.detect_arch(model_path, device)
        if model_arch is None:
            # Model architecture not found
            # model_arch, model_obj = fwk.find_model_arch(model_path, device)
            warn(f"{red("[E] Erroneous model or unsupported architecture:")}: {model_path}")
            return None
        # nnlogger.debug(yellow(f"fwk={fwk.type.value}, arch={model_arch.name}"))

        model = self._create_model(
            nn_model_path=model_path,
            framework=fwk,
            model_arch=model_arch,
            model_obj=model_obj,
            device=device
        )

        try:
            resave = model.resave
        except:
            resave = False
        resave = False
        if resave:
            nnlogger.debug("resave")
            state_dict = torch.load(
                model_path,
                map_location='cpu',
                pickle_module=RestrictedUnpickle,
            )

            metadata = {
                # 'by': 'Herlegon',
                # 'pro': 1 if model.pro else 0,
                # 'denoise': model.denoise,
                'datetime': datetime.strptime(
                    time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    '%Y-%m-%dT%H:%M:%S%z').isoformat(),
                'an_int': 3,
                'a_tuple': (2,3,4),
                'a_dict': dict(a=1,b=2,c=3)
            }
            print(json.dumps(metadata))
            try:
                del state_dict['pro']
            except:
                pass
            state_dict[f'metadata'] = {}
            for k, v in metadata.items():
                state_dict[f'metadata'][k] = json.dumps(v)
                # print(json.dumps(v))

            metadata = {
                'datetime': datetime.strptime(
                    time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    '%Y-%m-%dT%H:%M:%S%z').isoformat(),
                'an_int': 3,
                'a_tuple': (2,3,4),
                'a_dict': dict(a=1,b=2,c=3)
            }
            state_dict[f'metadata'] = json.dumps(metadata)

            torch.save(state_dict,
                model_path.replace(".pth", "_rlg.pth")
            )

        if any(x <= 0 for x in (model.scale, model.in_nc, model.out_nc)):
            nnlogger.debug("warning: at least a property has not been found, unsupported model")
            # return None

        return model


    def session(self, model: NnModel) -> NnModelSession:
        """Returns an inference session for a model"""
        create_session_fct = model.arch.create_session
        if create_session_fct is not None:
            session: NnModelSession = create_session_fct(model)
        else:
            raise NotImplementedError(f"Cannot create session for {model.fwk_type.value}")

        return session


    @staticmethod
    def _create_model(
        nn_model_path:str,
        framework: NnFramework,
        model_arch: NnArchitecture,
        model_obj: NnModelObject,
        device: str = 'cpu'
    ) -> NnModel:

        if framework.type == NnFrameworkType.PYTORCH:
            model = PytorchModel(
                filepath=nn_model_path,
                framework=framework,
                arch=model_arch,
                state_dict=model_obj,
            )
        elif framework.type == NnFrameworkType.ONNX:
            model = OnnxModel(
                filepath=nn_model_path,
                framework=framework,
                arch=model_arch,
                model_proto=model_obj,
            )
        elif framework.type == NnFrameworkType.TENSORRT:
            model = TrtModel(
                filepath=nn_model_path,
                framework=framework,
                arch=model_arch,
                engine=model_obj,
            )
            if not device.startswith("cuda"):
                nnlogger.debug("[W] wrong device to load a tensorRT model, use default cuda device")
                device = "cuda:0"
            model.device = device
        else:
            raise ValueError("[E] Unknown framework")

        # Parse a model object
        model.arch_name = model_arch.name
        # TODO: put the following code in an Try-Except block
        # try:
        model_arch.parse(model)
        # except:
        #     pass
        return model


    def convert_to_onnx(
        self,
        model: NnModel,
        opset: int = 17,
        dtype: Idtype = 'fp32',
        static: bool = False,
        device: str = 'cpu',
        out_dir: str | Path | None = None,
        suffix: str | None = None,
    ) -> OnnxModel:
        """Convert a model into an onnx model.

        Args:
            model: input model
            opset: onnx opset version
            dtype: Idtype
            device: device used for this conversion. the converted model will not use fp16
                    if this device does not support it.
            outdir: directory to save the onnx model. If set to None,
                    the model is not saved
        """
        onnx_model: OnnxModel = None

        if model.fwk_type == NnFrameworkType.ONNX:
            nnlogger.debug(f"This model is already an ONNX model")
            return model

        # TODO: put the following code in an Try-Except block
        nnlogger.debug(yellow(f"[I] Convert to onnx model: device={device}, dtype={dtype}, opset={opset}"))
        if (
            model.arch is not None
            and (convert_fct := model.arch.to_onnx) is not None
        ):
            onnx_model_object: onnx.ModelProto = convert_fct(
                model=model,
                dtype=dtype,
                static=static,
                opset=opset,
                device=device,
            )
        else:
            raise RuntimeError(f"Cannot convert from {model.arch_name} to ONNX (unsupported)")

        # Instantiate a new model
        onnx_fwk = self.frameworks[NnFrameworkType.ONNX]
        model_arch, _ = onnx_fwk.detect_arch(onnx_model_object)
        onnx_model = self._create_model(
            nn_model_path='',
            framework=onnx_fwk,
            model_arch=model_arch,
            model_obj=onnx_model_object,
        )
        onnx_model.opset = opset

        # TODO: clean this
        onnx_dtype = model.dtypes[0]
        if onnx_dtype == 'fp16' and onnx_dtype in model.arch.dtypes:
            onnx_model.dtypes = set(['fp16'])
        elif onnx_dtype == 'bf16' and onnx_dtype in model.arch.dtypes:
            onnx_model.dtypes = set(['bf16'])
        else:
            onnx_model.dtypes = set(['fp32'])

        # Add some info (metadata)
        onnx_model.alt_arch_name = model.arch_name
        if onnx_model.scale == 0:
            onnx_model.scale = model.scale

        # Save this model
        if out_dir is not None:
            basename = os_path_basename(model.filepath)
            success = onnx_fwk.save(onnx_model, out_dir, basename, suffix)
            if success:
                nnlogger.debug(f"[I] Onnx model saved as {onnx_model.filepath}")
            else:
                nnlogger.debug(f"[E] Failed to save the Onnx model as {onnx_model.filepath}")

        return onnx_model


    def convert_to_tensorrt(self,
        model: NnModel,
        shape_strategy: ShapeStrategy,
        dtype: Idtype = '',
        optimization_level: int | None = None,
        opset: int = 17,
        device: str = "cuda:0",
        out_dir: str | Path | None = None,
        suffix: str | None = None,
    ) -> TrtModel:
        """Convert a model into a tensorrt model.
        Returns a new instance of model.
        Refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec

        Args:
            model: input model
            shape_strategy: specify the min/opt/max shapes.
                when static flag is set to True, the converter uses the opt shape
            dtype: datatype of the tensorRT engine, the onnx model will always be in fp32
            optimization_level: (not supported) Set the builder optimization level to build the engine with.
            opset: onnx opset version if the input model is NOT an onnx model.
            device: GPU device used for this conversion: This model shall run on the same
                device.
            out_dir: directory to save the onnx/tensorrt model. Not saved if set to None
            suffix: a suffix added to the model filename
        """

        if model.fwk_type == NnFrameworkType.TENSORRT:
            raise ValueError(f"This model is already a TensorRT model")

        trt_dtypes = set(['fp32'])
        if trt_dtypes:
            trt_dtypes.add(dtype)

        # Remove suffixes from ONNX basename
        basename = os_path_basename(model.filepath)
        if model.fwk_type == NnFrameworkType.ONNX:
            opset = model.opset
            basename = re.sub(r"_op\d{1,2}", '', basename)
            for dt in ('_fp32', '_fp16', '_bf16'):
                basename = basename.replace(dt, '')

        # Verify if tensor engine already exists, create a fake model
        if out_dir is not None:
            _model: TrtModel = TrtModel(
                framework=self.frameworks[NnFrameworkType.TENSORRT],
                arch=NnTensorrtArchitecture,
                arch_name='generic',
                filepath='',
                device=device,
                dtypes=trt_dtypes,
                engine=None,
                shape_strategy=shape_strategy,
                opset=opset,
            )
            trt_basename: str = generate_tensorrt_basename(
                _model, basename,
            )
            suffix = suffix if suffix is not None else ''
            filepath = os.path.join(out_dir, f"{trt_basename}{suffix}.engine")
            if os.path.exists(filepath):
                nnlogger.debug(f"[I] Engine {filepath} already exists, do not convert")
                return self.open(filepath, device)
            else:
                nnlogger.debug(f"[I] Engine {filepath} does not exist")
            del _model

        # Convert to Onnx
        nnlogger.debug(yellow(f"[I] Convert to onnx (fp32, {dtype})"))
        onnx_model: OnnxModel = self.convert_to_onnx(
            model=model,
            opset=opset,
            dtype='fp32',
            static=shape_strategy.static,
            device=device,
            out_dir=out_dir,
        )

        convert_to_tensorrt_fct = onnx_model.arch.to_tensorrt
        # TODO: put the following code in an Try-Except block
        trt_engine = None
        if convert_to_tensorrt_fct is not None:
            trt_engine = convert_to_tensorrt_fct(
                model=onnx_model,
                device=device,
                dtypes=trt_dtypes,
                shape_strategy=shape_strategy
            )
        else:
            raise NotImplementedError("Cannot convert to TensorRT")

        if trt_engine is None:
            nnlogger.debug(f"Error while converting {model.fwk_type} to TensorRT")
            return None

        # Instantiate a new model
        trt_fwk = self.frameworks[NnFrameworkType.TENSORRT]
        model_arch, _ = trt_fwk.detect_arch(trt_engine)
        trt_model = self._create_model(
            nn_model_path='',
            framework=trt_fwk,
            model_arch=model_arch,
            model_obj=trt_engine,
        )

        # Add specific params
        trt_model.opset = opset
        trt_model.shape_strategy = shape_strategy
        trt_model.dtypes.add('fp16')
        trt_model.scale = model.scale
        trt_model.dtypes = trt_dtypes.copy()

        # Save this engine as a model
        if out_dir is not None:
            nnlogger.debug(f"[V] save tensort RT engine to {out_dir}")
            success = trt_fwk.save(trt_model, out_dir, basename, suffix)
            if success:
                nnlogger.debug(f"[I] TRT engine saved as {trt_model.filepath}")
            else:
                nnlogger.debug(f"[E] Failed to save the TRT engine as {trt_model.filepath}")

        return trt_model


    def set_session_constructor(
        self,
        framework: NnFrameworkType,
        ModelSession: NnModelSession
    ):
        """Set a custom session contructor ffor a framework"""
        if framework == NnFrameworkType.TENSORRT and not is_tensorrt_available():
            raise ValueError("[E] Framework not supported: cannot set a custom session function")
        self.frameworks[framework].Session = ModelSession


nn_lib: NnLib = NnLib()


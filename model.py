from __future__ import annotations
from dataclasses import dataclass, field
import onnx
from typing import Any, TYPE_CHECKING

from .utils import arg_list

from .nn_pytorch.torch_types import (
    StateDict,
    TorchNnModule,
)
from .nn_tensor_rt.trt_types import (
    ShapeStrategy,
    TrtEngine,
)
from .nn_types import NnFrameworkType
if TYPE_CHECKING:
    from .architecture import NnArchitecture, SizeConstraint
    from .nn_types import NnModelDtype, NnFrameworkType
    from .framework import NnFramework



@dataclass
class SizeConstraint:
    min: tuple[int, int] | None = None
    max: tuple[int, int] | None = None
    modulo: int = 1

    def is_size_valid(
        self,
        size_or_shape: tuple[int, int, int] | tuple[int, int],
        is_shape: bool = True
    ) -> bool:
        """Return True if the size is valid.
        The size can be provided as a np.shape (h,w,c) or as a tuple of dims (w, h)
        TODO: verify modulo
        """
        if is_shape:
            h, w = size_or_shape[:2]
        else:
            w, h = size_or_shape[:2]
        if self.min is not None:
            if w < self.min[0] or h < self.min[1]:
                return False
        if self.max is not None:
            if w > self.max[0] or h > self.max[1]:
                return False
        return True



@dataclass(slots=True)
class ModelExecutor:
    """Used for inference in a mp"""
    device: str
    fp16: bool
    pinned_mem: bool
    in_max_shape: tuple[int, int, int]



@dataclass
class GenericModel:
    """
        - (not supported)   model.to_tensorrt
        -  use              nnlib.to_onnx(model)
    """
    framework: NnFramework
    arch: NnArchitecture
    # Use a variable as this name may differ from the
    #   default arch name. This variable may be
    #   updated when parsing the model
    arch_name: str = 'unknown'

    scale: int = 0
    in_nc: int = 0
    out_nc: int = 0

    filepath: str | None = None

    # Use this device to parse a TensorRT model
    # TBD: for inference?
    device: str = 'cpu'

    # tensorRT and Onnx only: indicates which dtype
    # is supported by this model.
    # PyTorch: use the dtypes specified by each arch
    # TODO: move this as a property like size_constraint
    dtypes: set[NnModelDtype] = field(default_factory=set)

    # Object used to initialize an executor.
    # when in multiprocess. Useless otherwise
    executor: ModelExecutor | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    # Custom size constraints
    _size_constraint: SizeConstraint | None = None


    @property
    def fwk_type(self) -> NnFrameworkType:
        return self.framework.type


    def update(self, **kwargs):
        """Update multiple fields of this class.
            \nRaises an exception if a key is not defined.
        """
        defined_keys: list[str] = list(self.__dict__.keys())

        # Append the list of arguments used by a PyTorch nn.Module
        module_key: str = 'ModuleClass'
        if module_key in defined_keys:
            module = (
                kwargs[module_key]
                if module_key in kwargs
                else getattr(self, module_key, None)
            )
            if module is not None:
                defined_keys.extend(arg_list(kwargs['ModuleClass']))

        for key, value in kwargs.items():
            if key not in defined_keys:
                raise KeyError(f"Undefined key \'{key}\' in {self.__class__.__name__}")
            setattr(self, key, value)


    def __str__(self) -> str:
        class_str = f"{self.__class__}: {'{'}\n"
        for k, v in self.__dict__.items():
            if k in ['model_proto', 'state_dict', 'engine', 'arch', 'framework']:
                class_str += (
                    f"\t{k}: {f'{type(v).__name__} ...' if v is not None else 'None'}\n"
                )
                continue
            class_str += f"\t{k}: {type(v).__name__} = {v}\n"
        class_str += "}\n"
        return class_str


    def supported_dtypes(self) -> set[NnModelDtype]:
        """Returns supported dtypes by this model
        """
        return (
            self.arch.dtypes
            if self.arch.type != NnFrameworkType.TENSORRT
            else self.dtypes
        )

    @property
    def size_constraint(self) -> SizeConstraint:
        return (
            self.arch.size_constraint
            if self._size_constraint is None
            else self._size_constraint
        )

    @size_constraint.setter
    def size_constraint(self, size_constraint: SizeConstraint) -> None:
        self._size_constraint = size_constraint


    def is_size_valid(
        self,
        size_or_shape: tuple[int, int, int] | tuple[int, int],
        is_shape: bool = True
    ) -> bool:
        """Return True if the size is valid.
        The size can be provided as a np.shape (h,w,c) or as a tuple of dims (w, h)
        TODO: verify modulo
        """
        return (
            self.size_constraint.is_size_valid(size_or_shape, is_shape)
            if self.size_constraint is not None
            else True
        )



@dataclass
class OnnxModel(GenericModel):
    model_proto: onnx.ModelProto | None = None
    opset: int | None = None
    alt_arch_name: str = ''
    in_shape_order: str = 'NCHW'



@dataclass
class PytorchModel(GenericModel):
    state_dict: StateDict = field(default_factory=StateDict)
    num_feat: int = 0
    num_conv: int = 0
    ModuleClass: TorchNnModule | None = None
    module: TorchNnModule | None = None



@dataclass
class TrtModel(GenericModel):
    engine: TrtEngine = None
    # profiles: list[ShapeStrategy] = field(default_factory=[ShapeStrategy])
    shape_strategy: ShapeStrategy = field(default_factory=ShapeStrategy)
    opset: int | None = None
    # Put here the device id? no? used for conversion load
    device: str = ""


NnModel = OnnxModel | PytorchModel | TrtModel

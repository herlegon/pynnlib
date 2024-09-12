from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..import_libs import is_tensorrt_available
from ..logger import nnlogger
if is_tensorrt_available():
    from tensorrt import ICudaEngine
    TrtEngine: TypeAlias = ICudaEngine
else:
    nnlogger.debug("[W] Importing ICudaEngine failed")
    TrtEngine: TypeAlias = Any

@dataclass
class TensorrtModel:
    engine: TrtEngine

@dataclass
class ShapeStrategy:
    """Shapes: (width, height)
    """
    static: bool = False
    min_size: tuple[int, int] = (0, 0)
    opt_size: tuple[int, int] = (0, 0)
    max_size: tuple[int, int] = (0, 0)

    def is_valid(self) -> bool:
        for d in range(2):
            values = [x[d] for x in (self.min_size, self.opt_size, self.max_size)]
            if min([values[i+1] - values[i] for i in range(len(values)-1)]) < 0:
                return False
        return True

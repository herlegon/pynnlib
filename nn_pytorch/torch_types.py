from __future__ import annotations
from typing import Any, TypeVar

StateDict = dict[str, Any]

try:
    from torch import nn
    TorchNnModule = TypeVar("TorchNnModule", bound=nn.Module)
except:
    TorchNnModule = TypeVar("TorchNnModule", bound=Any)

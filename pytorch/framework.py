from __future__ import annotations
from collections import OrderedDict
from .model import MODEL_ARCHITECTURES
from .model.parser import get_model_arch
from .inference.session import PyTorchSession
from pynnlib.framework import (
    NnFramework,
    NnFrameworkType,
)


FRAMEWORK: NnFramework = NnFramework(
    type=NnFrameworkType.PYTORCH,
    architectures=OrderedDict((a.name, a) for a in MODEL_ARCHITECTURES),
    get_arch=get_model_arch,
    Session=PyTorchSession,
)

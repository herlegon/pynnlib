from collections import OrderedDict
from .model.arch import MODEL_ARCHITECTURES
from .model.parser import get_model_arch
from .model.save import save
from .inference.session import TensorRtSession
from pynnlib.framework import (
    NnFramework,
    NnFrameworkType,
)

FRAMEWORK: NnFramework = NnFramework(
    type=NnFrameworkType.TENSORRT,
    architectures=OrderedDict((a.name, a) for a in MODEL_ARCHITECTURES),
    get_arch=get_model_arch,
    save=save,
    Session=TensorRtSession,
)

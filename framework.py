from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from pprint import pprint
import random
import re
import sys
from typing import OrderedDict

from .logger import nnlogger
from .import_libs import is_tensorrt_available
from .nn_types import NnFrameworkType, NnModelObject
from .model import NnModel
from .capabilities import get_system_capabilities
from .architecture import GetModelArchFct, NnArchitecture
from .session import NnModelSession
from .utils import swap_keys_values
from .utils.p_print import *



@dataclass
class NnFramework:
    """A framework contains helpers to instantiate a model,
    find a model arch, and create an inference session
    """
    type: NnFrameworkType
    architectures: OrderedDict[str, NnArchitecture]
    get_arch: GetModelArchFct | None = None
    save: Callable[[NnModel, Path | str, str, str], bool] | None = None
    Session: NnModelSession | None = None


    def detect_arch(
        self,
        model: NnModelObject,
        device: str = 'cpu'
    ) -> tuple[NnArchitecture, NnModelObject]:
        """Find the model architecture.
         may be a path or any model type of the fwks
        """
        try:
            return self.get_arch(model, self.architectures, device)
        except:
            nnlogger.error(f"[E] architecture not found for {model}")
        return (None, None)


    def add_model_arch(self, arch: NnArchitecture) -> None:
        if arch.name in self.architectures.keys():
            raise ValueError(f"Duplicated arch: {arch.name}")
        self.architectures[arch.name] = arch


    def create_session(self, model: NnModel) -> NnModelSession:
        try:
            return self.architectures[model.arch.name].create_session(model)
        except:
            pass
        raise NotImplementedError(f"Cannot create session for {model.filepath}")



def import_frameworks() -> dict[NnFrameworkType, NnFramework]:
    """This function walk through the arch directory and try to import frameworks"""
    frameworks: dict[NnFrameworkType, NnFramework] = {}

    supported_fwks: tuple[str] = (
        [c.value.lower() for c in get_system_capabilities().keys()]
    )
    nn_directory = Path(os.path.dirname(__file__)).absolute()
    nn_frameworks: list[str] = []
    for dir in os.listdir(nn_directory):
        if (
            not os.path.isdir(os.path.join(nn_directory, dir))
            or '__pycache__' in dir
        ):
            continue

        if re.subn('[-_ ]', '', dir.lower().replace('nn_', ''))[0] in supported_fwks:
            nn_frameworks.append(dir)
    nnlogger.debug(f"[V] Framework directories: {', '.join(supported_fwks)}")

    # Current module name
    fwk_module_base = os.path.basename(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)))
        )
    )

    # Walk trough subdirectories
    detected_fwks: list[str] = []
    # Randomize for debugging purpose
    random.shuffle(nn_frameworks)
    for nn_fwk in nn_frameworks:

        nn_fwk_dir = os.path.join(nn_directory, nn_fwk)
        if not os.path.exists(nn_fwk_dir) or not os.path.isdir(nn_fwk_dir):
            continue

        # Check structure
        is_candidate = True
        for folder in ('archs', 'inference'):
            if not os.path.isdir(os.path.join(nn_fwk_dir, folder)):
                is_candidate = False

        init_file = os.path.join(nn_fwk_dir, 'framework.py')
        if not os.path.isfile(init_file):
            is_candidate = False
        if not is_candidate:
            continue

        fwk_module_name = f"{fwk_module_base}.{nn_fwk}.framework".replace("/", ".")
        detected_fwks.append((fwk_module_name, init_file))

    # Load modules
    def add_framework(frameworks, fwk: NnFramework) -> None:
        """Add a framework to the database.
        """
        fwk_key: NnFrameworkType = fwk.type
        nnlogger.debug(f"[V] Adding framework {fwk_key}")
        if fwk_key in frameworks.keys():
            raise ValueError(f"Framework {fwk_key} already added. Cannot overwrite existing.")
        frameworks[fwk_key] = fwk

    for fwk_module_name, init_file in detected_fwks:
        if (
            "tensorrt" in fwk_module_name.replace('_', '')
            and not is_tensorrt_available()
        ):
            continue

        is_already_loaded = False
        if fwk_module_name in sys.modules:
            # nnlogger.debug(f"{fwk_module_name!r} already in sys.modules")
            is_already_loaded = True
        elif (module_spec := importlib.util.spec_from_file_location(fwk_module_name, init_file)) is not None:
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[fwk_module_name] = module
            nnlogger.debug(f"[V] Importing framework: {fwk_module_name}")
            try:
                module_spec.loader.exec_module(module)
                nnlogger.debug(f"[I] Module {fwk_module_name} loaded")
            except Exception as e:
                module_spec.loader.exec_module(module)
                nnlogger.debug(f"[W] Failed to load module {fwk_module_name}")
        # else:
        #     nnlogger.debug(f"can't find {fwk_module_name!r} module")

        module = sys.modules[fwk_module_name]
        if (hasattr(module, "FRAMEWORK")
            and getattr(module, "FRAMEWORK") is not None):
            fwk: NnFramework = module.FRAMEWORK
            add_framework(frameworks=frameworks, fwk=fwk)

        elif not is_already_loaded:
            del sys.modules[fwk_module_name]

    return frameworks




framework_to_extensions: dict[NnFrameworkType, tuple[str]] = {
    NnFrameworkType.ONNX : ['.onnx'],
    NnFrameworkType.PYTORCH : ['.pt', '.pth', '.ckpt', '.safetensors'],
    NnFrameworkType.TENSORRT : ['.engine', '.trt'],
}
extensions_to_framework: dict[str, NnFrameworkType] = swap_keys_values(framework_to_extensions)

def get_supported_model_extensions(framework: NnFrameworkType | None = None) -> tuple[int]:
    if framework is None:
        return tuple(extensions_to_framework.keys())
    return tuple(framework_to_extensions[framework])

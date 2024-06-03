
from __future__ import annotations
from functools import partial
import importlib
import os
from pathlib import Path, PurePath
import sys
from typing import Type

from ..torch_types import (
    StateDict,
)
from pynnlib.logger import nnlogger
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.utils.p_print import *
from pynnlib.model import PytorchModel
from .helpers import parameters_to_args


def import_model_architectures() -> list[NnPytorchArchitecture]:
    imported_archs: list[NnPytorchArchitecture] = []

    # Current module name
    dir_parts: tuple[str] = PurePath(
        os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    ).parts
    model_module_base: str = '.'.join(dir_parts[-3:])

    # Get all subfolders
    model_directory: Path = Path(os.path.dirname(__file__)).absolute()
    arch_dirs: set = set([d for d in os.listdir(model_directory) \
        if os.path.isdir(os.path.join(model_directory, d))])
    arch_dirs.discard('__pycache__')

    # Walk trough subdirectories
    detected_archs: list[tuple[str, str, str]] = []
    for arch in arch_dirs:
        # Check structure
        init_file: str = os.path.join(model_directory, arch, '__init__.py')
        if (not os.path.isfile(init_file)
            or not os.path.isdir(os.path.join(model_directory, arch, 'module'))):
            continue

        # Append to detected
        arch_module_name: str = f"{model_module_base}.{arch}"
        detected_archs.append((arch, arch_module_name, init_file))

    # Import modules
    for arch, arch_module_name, init_file in detected_archs:
        if arch_module_name in sys.modules:
            # nnlogger.debug(f"architecture {arch_module_name!r} already in sys.modules")
            module = sys.modules[arch_module_name]
        elif (module_spec := importlib.util.spec_from_file_location(arch_module_name, init_file)) is not None:
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[arch_module_name] = module
            # nnlogger.debug(f"load {arch_module_name!r} from {module}")
            module_spec.loader.exec_module(module)
            # nnlogger.debug(f"{arch_module_name!r} has been imported")
        # else:
        #     nnlogger.debug(f"can't find the {arch_module_name!r} module")

        # if (hasattr(module, "MODEL_ARCHITECTURES")
        #     and getattr(module, "MODEL_ARCHITECTURES") is not None):
        try:
            if getattr(module, "MODEL_ARCHITECTURES") is not None:
                imported_archs.extend(module.MODEL_ARCHITECTURES)
        except:
            pass
    nnlogger.debug("[V] Imported PyTorch archs")
    for arch in imported_archs:
        nnlogger.debug(f"[V]\t{arch.name}")

    return imported_archs


def contains_keys(state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
    return all(key in state_dict for key in keys)


def create_session(model: Type[PytorchModel]) -> Type[PytorchModel]:
    TorchModule = model.ModuleClass
    args = parameters_to_args(model, TorchModule)
    model.module = TorchModule(**args)
    return model.framework.Session(model)


# Detect and list model architectures
MODEL_ARCHITECTURES: list[NnPytorchArchitecture] = import_model_architectures()


# Append common variables for all architectures
for arch in MODEL_ARCHITECTURES:
    arch.type = arch.name
    arch.detect = partial(contains_keys, keys=arch.detection_keys)
    arch.create_session = create_session

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utility classes and functions."""

import ctypes
import importlib
import inspect
import numpy as np
import os
import sys
import types
import io
import pickle
import re
import warnings
from typing import Any
import torch
from torch import Tensor


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    # assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    # assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    # assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    # assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    # assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__



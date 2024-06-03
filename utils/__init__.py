import inspect
import os
from pathlib import Path
from typing import Literal, Type


def arg_list(model_class: type) -> list[str]:
    return [
        kw
        for kw in inspect.signature(model_class.__init__).parameters.keys()
    ][1:]


def swap_keys_values(d: Type[dict]) -> Type[dict]:
    """Swap keys/value of a dict.
    Warning, it owerwrites a key, value pair if already exists"""
    swapped = type(d)()
    for k, v in d.items():
        if isinstance(v, list):
            swapped.update({x: k for x in v})
        else:
            swapped[v] = k
    return swapped


pathAccess: dict = {
    'r': os.R_OK,
    'w': os.W_OK,
    'rw': os.O_RDWR,
}


def is_access_granted(path: str | Path, access: Literal['r', 'w', 'rw']):
    return os.access(str(path), mode=pathAccess[access])


def path_split(fp: str | Path) -> tuple[str, str, str]:
    """Returns the [directory, basename, extension]
    of the given filepath.
    """
    base, extension = os.path.splitext(str(fp))
    directory, basename = os.path.split(base)
    return directory, basename, extension.lower()


def os_path_basename(fp: str | Path) -> str:
    """Return the basename without extension"""
    return os.path.splitext(os.path.basename(fp))[0]


def get_extension(fp: str) -> str:
    """Returns the extensions in lower case"""
    return os.path.splitext(fp)[1].lower()


def absolute_path(path: str | Path) -> str:
    if path is not None and path != "":
        return os.path.abspath(os.path.expanduser(str(path)))
    return path


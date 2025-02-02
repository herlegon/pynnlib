import os
import glob
from pprint import pprint
import sys
from typing import Any, Literal
import warnings
import torch
import torch.utils.cpp_extension
import importlib
import hashlib

from utils.p_print import *



_mat_plugins: dict[str, Any] = {}

verbosity: Literal["none", "brief", "full"] = "full"


os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
def _find_compiler_bindir() -> str | None:
    patterns = (
        "C:\\Visual_Studio\\*\\Community\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64",
        "C:\\Visual_Studio *\\vc/bin",
        "C:\\Program Files\\Microsoft Visual Studio\\*\\Community\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64",
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None




def compile_plugin(
    module_name: str,
    sources: tuple[str],
    verbose: bool = False,
    **build_kwargs
):
    global _mat_plugins
    if module_name in _mat_plugins:
        try:
            module = importlib.import_module(module_name)
            sys.modules[module_name] = module
        except:
            warnings.warn(yellow(f"failed to load module {module_name}, compile"))

    _mat_plugins[module_name] = None

    # Can compile?
    compiler_dir: str | None = _find_compiler_bindir()
    if (
        not torch.cuda.is_available()
        or not compiler_dir
        or not sys.platform == "win32"
    ):
        return

    sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
    if verbose:
        print(f"Setting up PyTorch plugin \"{module_name}\"...")

    # Append compiler dir if cl.exe is not already in path
    if os.system("where cl.exe >nul 2>nul") != 0:
        compiler_bindir = _find_compiler_bindir()
        if compiler_bindir is None:
            raise RuntimeError(f"Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in \"{__file__}\".")
        os.environ["PATH"] += ";" + compiler_bindir
        if verbose:
            print(yellow("found compiler"), compiler_bindir, flush=True)

    # Torch build directory
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.abspath(os.path.dirname(__file__))
    torch_build_dir = torch.utils.cpp_extension._get_build_directory(
        module_name, verbose=verbose
    )

    # Compute hash to detect modifications
    hash_md5 = hashlib.md5()
    for src in sources:
        with open(src, "rb") as f:
            hash_md5.update(f.read())
    hash_file = os.path.join(torch_build_dir, hash_md5.hexdigest())

    # Recreate a directory to rebuild the module
    if not os.path.isfile(hash_file):
        # Recreate a directory
        try:
            os.rmdir(torch_build_dir)
        except:
            pass
        os.makedirs(torch_build_dir, exist_ok=True)
        with open(hash_file, 'w'):
            pass

    if verbose:
        print(lightcyan(f"Load cpp extension: {module_name}\nsources: {sources}"))
    torch.utils.cpp_extension.load(
        name=module_name,
        build_directory=torch_build_dir,
        verbose=verbose,
        sources=sources,
        **build_kwargs
    )

    module = importlib.import_module(module_name)
    _mat_plugins[module_name] = module

    return module

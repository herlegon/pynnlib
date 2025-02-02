import os
from pprint import pprint
import sys
import time
from typing import Any
import warnings
import torch
import torch.utils.cpp_extension
import importlib
import hashlib

from ...helpers import find_compiler_bindir
from utils.p_print import *


_lamnet_cuda_ext: dict[str, Any] = {}


def compile_cuda_ext(
    module_name: str,
    sources: tuple[str],
    verbose: bool = False,
    **build_kwargs
):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

    global _lamnet_cuda_ext
    if module_name in _lamnet_cuda_ext:
        try:
            module = importlib.import_module(module_name)
            sys.modules[module_name] = module
        except:
            warnings.warn(yellow(f"failed to load module {module_name}, compile"))

    _lamnet_cuda_ext[module_name] = None

    # Can compile?
    compiler_dir: str | None = find_compiler_bindir()
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
        if compiler_dir is None:
            raise RuntimeError(f"Could not find MSVC/GCC/CLANG installation on this computer. Check compiler_dir()")
        os.environ["PATH"] += ";" + compiler_dir
        if verbose:
            print(yellow("found compiler"), compiler_dir, flush=True)

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
    do_compile: bool = False
    if not os.path.isfile(hash_file):
        # Recreate a directory
        try:
            os.rmdir(torch_build_dir)
        except:
            pass
        os.makedirs(torch_build_dir, exist_ok=True)
        with open(hash_file, 'w'):
            pass
        do_compile = True

    if verbose:
        print(lightcyan(f"Load cpp extension: {module_name}\nsources: {sources}"))
    start_time: float = time.time()
    torch.utils.cpp_extension.load(
        name=module_name,
        build_directory=torch_build_dir,
        verbose=verbose or do_compile,
        sources=sources,
        with_cuda=True,
        **build_kwargs
    )
    print(f"LAMNet: {'compiled' if do_compile else 'loaded'} {module_name} in {time.time() - start_time:.02f}s")

    module = importlib.import_module(module_name)
    _lamnet_cuda_ext[module_name] = module

    return module


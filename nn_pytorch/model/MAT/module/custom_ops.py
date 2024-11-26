# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
from pprint import pprint
import sys
import torch
import torch.utils.cpp_extension
import importlib
import hashlib
import shutil
from pathlib import Path

from torch.utils.file_baton import FileBaton

from utils.p_print import *

#----------------------------------------------------------------------------
# Global options.

verbosity = 'full' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:\\Visual_Studio\\*\\Community\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64',
        'C:\\Visual_Studio *\\vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    modules = {
        'bias_act_plugin': "bias_act_plugin",
        'upfirdn2d_plugin': "upfirdn2d_plugin",
    }
    for _name, _dir in modules.items():
        sys.path.append(_dir)
        if os.path.exists(_dir):
           _cached_plugins[_name] = importlib.import_module(_name)


    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    print()

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Compile and load.
        verbose_build = (verbosity == 'full')
        print("found compiler\n", flush=True)

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        source_dirs_set = set(os.path.dirname(source) for source in sources)
        print(source_dirs_set)
        print()

        # os.environ['TORCH_EXTENSIONS_DIR'] = os.path.abspath(os.path.dirname(__file__))
        # print(f"env: {os.environ['TORCH_EXTENSIONS_DIR']}")

        if len(source_dirs_set) == 1 and ('TORCH_EXTENSIONS_DIR' in os.environ):
            all_source_files = sorted(list(x for x in Path(list(source_dirs_set)[0]).iterdir() if x.is_file()))
            print("\nall source files")
            print(all_source_files)

            # Compute a combined hash digest for all source files in the same
            # custom op directory (usually .cu, .cpp, .py and .h files).
            hash_md5 = hashlib.md5()
            print(yellow("hash_md5"), flush=True)
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            build_dir = torch.utils.cpp_extension._get_build_directory(
                module_name, verbose=verbose_build
            )
            print(yellow(f"build dir: {build_dir}"))
            digest_build_dir = os.path.join(build_dir, hash_md5.hexdigest())

            if not os.path.isdir(digest_build_dir):
                os.makedirs(digest_build_dir, exist_ok=True)
                print(yellow("baton"), flush=True)
                baton = FileBaton(os.path.join(digest_build_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        for src in all_source_files:
                            shutil.copyfile(src, os.path.join(digest_build_dir, os.path.basename(src)))
                    finally:
                        baton.release()
                else:
                    # Someone else is copying source files under the digest dir,
                    # wait until done and continue.
                    baton.wait()
            digest_sources = [os.path.join(digest_build_dir, os.path.basename(x)) for x in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,
                verbose=verbose_build, sources=digest_sources, **build_kwargs)
        else:
            print(yellow("torch.utils.cpp_extension.load"))
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                **build_kwargs
            )
        print(f"load {module_name}")
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------

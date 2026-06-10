"""torch-amgx build via torch.utils.cpp_extension.

The C++ extension links against AmgX (vendored in third_party/AMGX as a
submodule, built once via the project-level CMakeLists.txt). At wheel-
build time, the CI bundles the AmgX shared library (libamgxsh.so /
amgxsh.dll) into the wheel via auditwheel / delvewheel; here we just
need the build to find it at compile time.

For development installs (`pip install -e .`) the build pipeline is:

  1. cmake -S third_party/AMGX -B build/amgx (configures AmgX)
  2. cmake --build build/amgx --target amgxsh (builds AmgX shared lib)
  3. cmake --install build/amgx --prefix build/amgx/install
  4. torch.utils.cpp_extension picks up `-Lbuild/amgx/install/lib -lamgxsh`

The dev install path is documented in README; the wheel CI does all
four steps automatically inside the matrix job.
"""
from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).parent.resolve()
AMGX_DIR = Path(os.environ.get("AMGX_DIR",
                               HERE / "build" / "amgx" / "install"))


def _amgx_paths():
    """Locate AmgX headers, lib, and shared lib produced by CMake."""
    include = AMGX_DIR / "include"
    lib_dir = AMGX_DIR / "lib"
    # Library name is platform-dependent
    if os.name == "nt":
        lib_name = "amgxsh"
        runtime_lib = lib_dir / "amgxsh.dll"
    else:
        lib_name = "amgxsh"  # libamgxsh.so on Linux; ld -l strips lib/.so
        runtime_lib = lib_dir / "libamgxsh.so"

    if not include.exists():
        raise RuntimeError(
            f"AmgX include directory not found at {include}.\n"
            "Either build AmgX first (see scripts/build_amgx.sh) or set "
            "the AMGX_DIR env var to point at an existing AmgX install."
        )
    return include, lib_dir, lib_name, runtime_lib


_inc, _libdir, _libname, _runtime = _amgx_paths()


extension = CUDAExtension(
    name="torch_amgx._C",
    sources=[
        "csrc/amgx_solver.cu",
        "csrc/bindings.cpp",
    ],
    include_dirs=[str(_inc), "csrc"],
    library_dirs=[str(_libdir)],
    libraries=[_libname],
    extra_compile_args=(
        {
            # POSIX (Linux WSL etc.)
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "-fPIC"],
        }
        if os.name != "nt" else
        {
            # MSVC: /permissive- enforces strict standards conformance,
            # which torch 2.11's compiled_autograd.h requires to dodge
            # the C2872 'std': ambiguous-symbol error in MSVC's default
            # permissive mode. /Zc:__cplusplus exposes the real
            # __cplusplus macro so torch's header conditionals pick the
            # C++17 path.
            "cxx": ["/O2", "/std:c++17", "/permissive-", "/Zc:__cplusplus",
                    "/EHsc"],
            # nvcc -Xcompiler passes flags through to cl.exe
            "nvcc": ["-O3", "--use_fast_math",
                     "-Xcompiler", "/permissive-",
                     "-Xcompiler", "/Zc:__cplusplus",
                     "-Xcompiler", "/EHsc"],
        }
    ),
)


setup(
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)

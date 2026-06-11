"""torch-amgx -- PyTorch-native binding for NVIDIA AmgX.

Re-exports the user-facing API:

* :func:`solve_csr` / :func:`solve_coo` -- one-shot solves with autograd.
* :class:`Solver` -- reusable solver wrapping :class:`AmgXSolver`.
* :class:`Config` -- frozen dataclass describing the AmgX method.
* :class:`SolveInfo` -- returned alongside ``x`` when ``return_info=True``.

The low-level :class:`AmgXSolver` C++ class is also exposed via
``torch_amgx._C.AmgXSolver`` for power users.
"""
from __future__ import annotations

import os as _os
import sys as _sys


def _setup_dll_search_path():
    """On Windows, Python 3.8+ ignores PATH for DLLs loaded by ``.pyd``
    extensions. We have to register the directory holding AmgX's shared
    library via :func:`os.add_dll_directory` before importing ``_C``.

    Resolution order:

    1. ``TORCH_AMGX_LIB_DIR`` env var -- explicit override for development
       builds or unusual deploys.
    2. Bundled ``torch_amgx.libs/`` next to this package -- the layout
       :command:`delvewheel repair` produces in wheel builds.
    3. ``<package>/_libs/`` -- a developer-friendly alternative location
       for ``pip install -e .`` builds.
    4. (Non-Windows): no-op; rpaths embedded by ``auditwheel repair``
       handle the lookup automatically on Linux.
    """
    if _sys.platform != "win32":
        return
    _here = _os.path.dirname(_os.path.abspath(__file__))
    candidates = []
    env_dir = _os.environ.get("TORCH_AMGX_LIB_DIR")
    if env_dir:
        candidates.append(env_dir)
    # delvewheel default layout
    candidates.append(_os.path.join(_os.path.dirname(_here), "torch_amgx.libs"))
    # dev-install fallback
    candidates.append(_os.path.join(_here, "_libs"))
    for path in candidates:
        if path and _os.path.isdir(path):
            _os.add_dll_directory(path)


_setup_dll_search_path()

from .solver import (
    Config,
    SolveInfo,
    Solver,
    solve_csr,
    solve_coo,
    is_available,
    amgx_version,
)

__all__ = [
    "Config",
    "SolveInfo",
    "Solver",
    "solve_csr",
    "solve_coo",
    "is_available",
    "amgx_version",
]
__version__ = "0.1.0a1"

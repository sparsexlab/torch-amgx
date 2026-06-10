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
__version__ = "0.1.0"

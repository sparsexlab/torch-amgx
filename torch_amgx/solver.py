"""torch-amgx Python facade -- dataclass-based config + high-level Solver.

Note: torch-amgx is the binding layer only. To use it with autograd,
wrap the forward solve in your own torch.autograd.Function (this is
what torch-sla does in its amgx backend).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch


# Lazy C extension import; it pulls in CUDA + AmgX shared lib, which we
# don't want to require at module import time on non-GPU machines.
_C = None


def _load_c():
    global _C
    if _C is None:
        try:
            from . import _C as _native  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "torch_amgx._C native extension is not available. "
                "Install a prebuilt wheel (`pip install torch-amgx`) "
                "or build from source -- see the README."
            ) from exc
        _C = _native
    return _C


def is_available() -> bool:
    """Return ``True`` iff the C extension can be loaded AND CUDA is
    available on the current torch install."""
    if not torch.cuda.is_available():
        return False
    try:
        _load_c()
        return True
    except ImportError:
        return False


def amgx_version() -> str:
    """Return the AmgX runtime API version, e.g. ``\"2.5\"``."""
    return _load_c().amgx_version()


# ====================================================================== #
# Public dataclasses
# ====================================================================== #
MethodName = Literal["auto", "amg", "pcg", "cg", "pbicgstab", "bicgstab",
                     "fgmres", "gmres"]


@dataclass(frozen=True)
class Config:
    """Frozen dataclass describing an AmgX configuration.

    Set ``amgx_config_str`` to pass a literal AmgX printf-style config
    string and short-circuit method-based construction.

    Examples
    --------
    >>> Config(method="pbicgstab", tol=1e-8, maxiter=200)
    >>> Config(amgx_config_str="config_version=2,solver(main)=AMG,...")
    """
    method: MethodName = "auto"
    tol: float = 1e-8
    maxiter: int = 200
    presweeps: int = 1
    postsweeps: int = 1
    amgx_config_str: Optional[str] = None  # explicit override

    def build_config_str(self) -> str:
        """Materialize the printf-style AmgX config string."""
        if self.amgx_config_str is not None:
            return self.amgx_config_str

        method = "pbicgstab" if self.method == "auto" else self.method
        method = method.lower()

        if method == "amg":
            outer_block = (
                "solver(main)=AMG,"
                f"main:max_iters={self.maxiter},"
                f"main:tolerance={self.tol},"
                "main:convergence=ABSOLUTE,"
                "main:norm=L2,"
                "main:cycle=V,"
                f"main:presweeps={self.presweeps},"
                f"main:postsweeps={self.postsweeps},"
                "main:monitor_residual=1,"
                "main:print_solve_stats=0"
            )
            return "config_version=2," + outer_block

        outer_solver_map = {
            "pcg": "PCG", "cg": "PCG",
            "pbicgstab": "PBICGSTAB", "bicgstab": "PBICGSTAB",
            "fgmres": "FGMRES", "gmres": "FGMRES",
        }
        if method not in outer_solver_map:
            raise ValueError(
                f"Unknown AmgX method {method!r}; expected one of "
                f"auto, amg, cg, pcg, bicgstab, pbicgstab, gmres, fgmres."
            )
        outer = outer_solver_map[method]
        return (
            "config_version=2,"
            f"solver(main)={outer},"
            f"main:max_iters={self.maxiter},"
            f"main:tolerance={self.tol},"
            "main:convergence=ABSOLUTE,"
            "main:norm=L2,"
            "main:monitor_residual=1,"
            "main:print_solve_stats=0,"
            "main:preconditioner(amg)=AMG,"
            "amg:max_iters=1,"
            "amg:cycle=V,"
            f"amg:presweeps={self.presweeps},"
            f"amg:postsweeps={self.postsweeps}"
        )


@dataclass
class SolveInfo:
    """Diagnostics returned alongside ``x`` when ``return_info=True``."""
    iter_count: int = 0
    residual: float = float("nan")
    converged: bool = False
    method: str = ""


# ====================================================================== #
# Solver: explicit-lifecycle reusable solver
# ====================================================================== #
class Solver:
    """Reusable AmgX solver.

    One instance owns a (Config + Resources + Matrix + Solver) graph on
    a single CUDA device. :meth:`setup_csr` uploads a matrix and runs
    AmgX's setup phase; :meth:`solve` runs the iteration loop against a
    right-hand side. Repeated solves on the same setup are cheap.

    Lifecycle
    ---------
    Construct with a :class:`Config` and a CUDA device; the underlying
    AmgX handles are torn down when the Python object is garbage-
    collected.
    """

    def __init__(self,
                 config: Config,
                 device: Union[str, torch.device, None] = None):
        if device is None:
            device = torch.device("cuda", 0)
        elif isinstance(device, str):
            device = torch.device(device)
        if not device.type == "cuda":
            raise RuntimeError(
                f"torch_amgx.Solver requires a CUDA device; got {device}"
            )
        self._config = config
        self._cuda_device = device
        cfg_str = config.build_config_str()
        self._solver = _load_c().AmgXSolver(cfg_str, device)

    @property
    def config(self) -> Config: return self._config

    @property
    def device(self) -> torch.device: return self._cuda_device

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup_csr(self,
                  indptr: torch.Tensor,
                  indices: torch.Tensor,
                  values: torch.Tensor,
                  n: int) -> None:
        """Upload a CSR matrix and run AmgX setup."""
        self._solver.setup_csr(indptr, indices, values, n)

    def setup_coo(self,
                  val: torch.Tensor,
                  row: torch.Tensor,
                  col: torch.Tensor,
                  shape: Tuple[int, int]) -> None:
        """Convenience: assemble torch CSR on-device from COO, then setup."""
        n = shape[0]
        if shape[0] != shape[1]:
            raise ValueError("AmgX requires a square matrix")
        sparse_coo = torch.sparse_coo_tensor(
            torch.stack([row, col]), val, shape).coalesce()
        sparse_csr = sparse_coo.to_sparse_csr()
        self.setup_csr(
            sparse_csr.crow_indices(),
            sparse_csr.col_indices(),
            sparse_csr.values(),
            n,
        )

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self,
              b: torch.Tensor,
              *,
              return_info: bool = False
              ) -> Union[torch.Tensor, Tuple[torch.Tensor, SolveInfo]]:
        x = self._solver.solve(b)
        if return_info:
            info = SolveInfo(
                iter_count=self._solver.iter_count,
                residual=self._solver.residual,
                converged=self._solver.converged,
                method=self._config.method,
            )
            return x, info
        return x

    def solve_into(self, b: torch.Tensor, x: torch.Tensor) -> None:
        """In-place variant. ``x`` is updated in place; useful for warm
        starts and to avoid an output allocation."""
        self._solver.solve_into(b, x)


# ====================================================================== #
# One-shot functional solves
# ====================================================================== #
def solve_csr(indptr: torch.Tensor,
              indices: torch.Tensor,
              values: torch.Tensor,
              shape: Tuple[int, int],
              b: torch.Tensor,
              *,
              config: Optional[Config] = None,
              return_info: bool = False
              ) -> Union[torch.Tensor, Tuple[torch.Tensor, SolveInfo]]:
    """One-shot CSR solve via a transient :class:`Solver`.

    For repeated solves on the same matrix prefer the :class:`Solver`
    class -- this function pays the AmgX setup cost on every call.
    """
    if config is None:
        config = Config()
    n = shape[0]
    if shape[0] != shape[1]:
        raise ValueError("AmgX requires a square matrix")
    s = Solver(config, device=values.device)
    s.setup_csr(indptr, indices, values, n)
    return s.solve(b, return_info=return_info)


def solve_coo(val: torch.Tensor,
              row: torch.Tensor,
              col: torch.Tensor,
              shape: Tuple[int, int],
              b: torch.Tensor,
              *,
              config: Optional[Config] = None,
              return_info: bool = False
              ) -> Union[torch.Tensor, Tuple[torch.Tensor, SolveInfo]]:
    """One-shot COO solve. Internally converts to CSR on-device."""
    if config is None:
        config = Config()
    if shape[0] != shape[1]:
        raise ValueError("AmgX requires a square matrix")
    s = Solver(config, device=val.device)
    s.setup_coo(val, row, col, shape)
    return s.solve(b, return_info=return_info)

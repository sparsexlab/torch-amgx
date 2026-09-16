"""Several AmgX solvers may be alive at once.

AmgX's ``Resources`` destructor tears down process-global state -- the
cuSPARSE and cuBLAS handles and, via ``amgx::free_resources()``, the device
memory pools. torch-amgx used to create one ``Resources`` per solver, so
destroying any one of them pulled that shared state out from under every other
live solver: AmgX printed

    !!! detected some memory leaks in the code: trying to free non-empty
        temporary device pool !!!

and the next AmgX call died with ``Cuda failure: 'invalid argument'``, which
aborts the process (SIGABRT, exit 134). Two live solvers were enough. A solver
plus its adjoint is the normal case for autograd callers, so this took down
whole test sessions rather than just leaking at exit.

Resources is now shared per device, so these tests exercise the case that used
to abort. The abort happened at *destruction*, which is why each test forces
collection and then keeps solving: a stale shared handle shows up as a failure
on the surviving solver, not on the one that was dropped.
"""
from __future__ import annotations

import gc

import pytest
import torch

import torch_amgx


pytestmark = pytest.mark.skipif(
    not torch_amgx.is_available(),
    reason="torch-amgx requires CUDA + a working AmgX runtime",
)


def _tridiag_csr(n: int, device: torch.device, dtype=torch.float64):
    """1-D Poisson stencil in CSR, built on device."""
    indptr = torch.empty(n + 1, dtype=torch.int32, device=device)
    rows, cols, vals = [], [], []
    ptr = 0
    for i in range(n):
        indptr[i] = ptr
        for j, v in ((i - 1, -1.0), (i, 2.0), (i + 1, -1.0)):
            if 0 <= j < n:
                cols.append(j)
                vals.append(v)
                ptr += 1
        rows.append(i)
    indptr[n] = ptr
    return (indptr,
            torch.tensor(cols, dtype=torch.int32, device=device),
            torch.tensor(vals, dtype=dtype, device=device))


def _make_solver(n: int, device: torch.device) -> torch_amgx.Solver:
    indptr, indices, values = _tridiag_csr(n, device)
    s = torch_amgx.Solver(torch_amgx.Config(method="pbicgstab", tol=1e-10,
                                            maxiter=200))
    s.setup_csr(indptr, indices, values, n)
    return s


def _assert_solves(solver: torch_amgx.Solver, n: int, device: torch.device):
    b = torch.ones(n, dtype=torch.float64, device=device)
    x = solver.solve(b)
    assert torch.isfinite(x).all(), "solve returned non-finite values"
    return x


def test_two_solvers_coexist():
    """The minimal case that used to abort: two live solvers."""
    device = torch.device("cuda")
    a = _make_solver(64, device)
    b = _make_solver(96, device)
    _assert_solves(a, 64, device)
    _assert_solves(b, 96, device)


def test_dropping_one_solver_leaves_the_others_usable():
    """Destroying a solver must not invalidate its siblings.

    This is the exact shape of the old bug: the first destructor took down the
    shared cuSPARSE/cuBLAS handles and memory pools, so the *survivor* was the
    one that failed.
    """
    device = torch.device("cuda")
    keep = _make_solver(64, device)
    before = _assert_solves(keep, 64, device)

    for size in (80, 96, 112):
        doomed = _make_solver(size, device)
        _assert_solves(doomed, size, device)
        del doomed
        gc.collect()
        torch.cuda.synchronize()
        after = _assert_solves(keep, 64, device)
        torch.testing.assert_close(after, before)


def test_many_solvers_alive_at_once():
    """Hold more than a couple, then drop them all in one go."""
    device = torch.device("cuda")
    solvers = [(_make_solver(32 + 16 * i, device), 32 + 16 * i) for i in range(6)]
    for s, n in solvers:
        _assert_solves(s, n, device)
    survivor, survivor_n = solvers[0]
    del solvers[1:]
    gc.collect()
    torch.cuda.synchronize()
    _assert_solves(survivor, survivor_n, device)

"""End-to-end smoke tests for torch-amgx.

Skipped if either no CUDA is available or torch_amgx._C failed to load
(the package was installed without a working AmgX binary).
"""
from __future__ import annotations

import pytest
import torch

import torch_amgx


pytestmark = pytest.mark.skipif(
    not torch_amgx.is_available(),
    reason="torch-amgx requires CUDA + a working AmgX runtime",
)


def _poisson_2d_csr(n: int, device: torch.device, dtype=torch.float64):
    """Return (indptr, indices, values, shape) for the 5-point 2D
    Laplacian on an n×n grid -- pure-torch construction on device, no
    scipy / numpy round-trip."""
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j in range(n):
            r = i * n + j
            rows.append(r); cols.append(r); vals.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    rows.append(r)
                    cols.append(ni * n + nj)
                    vals.append(-1.0)
    row = torch.tensor(rows, dtype=torch.long, device=device)
    col = torch.tensor(cols, dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=dtype,    device=device)
    nn = n * n
    sp = torch.sparse_coo_tensor(
        torch.stack([row, col]), val, (nn, nn)).coalesce().to_sparse_csr()
    return sp.crow_indices(), sp.col_indices(), sp.values(), (nn, nn)


def test_module_version_reports_amgx_runtime():
    """``amgx_version`` returns a major.minor string from the live runtime."""
    s = torch_amgx.amgx_version()
    assert "." in s, f"unexpected version string {s!r}"


def test_solve_csr_one_shot_poisson():
    """One-shot ``solve_csr`` recovers a planted x on a 32×32 Poisson 2D."""
    device = torch.device("cuda", 0)
    indptr, indices, values, shape = _poisson_2d_csr(32, device)
    torch.manual_seed(0)
    x_ref = torch.randn(shape[0], device=device, dtype=torch.float64)
    A = torch.sparse_csr_tensor(indptr, indices, values, shape)
    b = (A @ x_ref).detach()

    x, info = torch_amgx.solve_csr(
        indptr, indices, values, shape, b,
        config=torch_amgx.Config(method="pbicgstab", tol=1e-9, maxiter=200),
        return_info=True,
    )
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert info.converged, info
    assert rel_err < 1e-6, f"rel_err {rel_err}"


def test_reusable_solver_amortises_setup():
    """A :class:`Solver` setup once + 3 solves only pays setup overhead
    on the first call; subsequent solves reuse the cached hierarchy."""
    device = torch.device("cuda", 0)
    indptr, indices, values, shape = _poisson_2d_csr(32, device)
    solver = torch_amgx.Solver(
        torch_amgx.Config(method="pbicgstab", tol=1e-9, maxiter=200),
        device=device,
    )
    solver.setup_csr(indptr, indices, values, shape[0])

    A = torch.sparse_csr_tensor(indptr, indices, values, shape)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        x_ref = torch.randn(shape[0], device=device, dtype=torch.float64)
        b = (A @ x_ref).detach()
        x = solver.solve(b)
        assert ((x - x_ref).norm() / x_ref.norm()).item() < 1e-6


@pytest.mark.parametrize("preconditioner", [
    "jacobi_l1", "block_jacobi", "multicolor_dilu", "chebyshev",
])
def test_non_amg_preconditioner_converges(preconditioner):
    """Smoke each non-default preconditioner choice end-to-end on the
    2D Poisson stencil; convergence rate varies but all of these must
    drive a 32x32 SPD system below 1e-4 within 1000 PCG iterations."""
    device = torch.device("cuda", 0)
    indptr, indices, values, shape = _poisson_2d_csr(32, device)
    solver = torch_amgx.Solver(
        torch_amgx.Config(
            method="pcg",
            preconditioner=preconditioner,
            tol=1e-8,
            maxiter=1000,
        ),
        device=device,
    )
    solver.setup_csr(indptr, indices, values, shape[0])

    A = torch.sparse_csr_tensor(indptr, indices, values, shape)
    torch.manual_seed(0)
    x_ref = torch.randn(shape[0], device=device, dtype=torch.float64)
    b = (A @ x_ref).detach()
    x = solver.solve(b)
    rel = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel < 1e-4, f"{preconditioner!r} stalled: rel-err {rel:.2e}"

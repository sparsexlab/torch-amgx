# torch-amgx

A **PyTorch-native** binding for [NVIDIA AmgX](https://github.com/NVIDIA/AMGX) — a GPU-resident sparse linear solver library providing algebraic multigrid (AMG) and Krylov methods (PCG, PBiCGStab, FGMRES, ...).

This package is **not a fork of `pyamgx`**. `pyamgx` is a numpy-shaped Cython wrapper that round-trips every tensor through host memory + scipy. `torch-amgx` is built from scratch as a PyTorch C++ extension:

* **Zero-copy**: torch CUDA tensors are passed directly to AmgX via `data_ptr()`. No numpy, no scipy.
* **Binding-only**: torch-amgx is the thin binding layer (PyTorch tensors <-> AmgX C API). Autograd (adjoint solve via conjugate transpose) lives in callers like `torch-sla` -- keeping that concern out of this repo means `torch-amgx` stays minimal and reusable for non-`torch-sla` consumers.
* **CUDA stream-aware**: respects `torch.cuda.current_stream()`.
* **Type-safe config**: `Config` is a frozen dataclass, not a printf-style string.
* **Clean lifecycle**: RAII via pybind11; no `STATUS_STACK_BUFFER_OVERRUN` at shutdown.

## Install

`torch-amgx` is **not on PyPI**. Prebuilt wheels are published to
[**GitHub Releases**](https://github.com/sparsexlab/torch-amgx/releases).

Each `(os, python)` ships **three CUDA variants** — cu12.4 / cu12.6 / cu12.8.
Because the binary is compiled against `torch`'s C++ ABI for a specific CUDA
toolkit, you **must pick the wheel whose CUDA matches your installed
`torch`'s CUDA** (check `python -c "import torch; print(torch.version.cuda)"`).
Installing a mismatched wheel fails at *import* time with a torch ABI error
(`ImportError: DLL load failed while importing _C` on Windows, or an
undefined-symbol error on Linux).

The CUDA variant is encoded in the wheel's **build tag** `0_cu124` /
`0_cu126` / `0_cu128` (right after the version), e.g.
`torch_amgx-<ver>-0_cu126-cp313-cp313-win_amd64.whl`. `pip install` its asset
URL directly:

```bash
# Example: Linux x86_64, Python 3.11, torch built for CUDA 12.6 -> 0_cu126
pip install https://github.com/sparsexlab/torch-amgx/releases/download/v0.1.0a11/torch_amgx-0.1.0a11-0_cu126-cp311-cp311-manylinux_2_35_x86_64.whl
```

```bash
# Example: Windows x64, Python 3.13, torch built for CUDA 12.4 -> 0_cu124
pip install https://github.com/sparsexlab/torch-amgx/releases/download/v0.1.0a11/torch_amgx-0.1.0a11-0_cu124-cp313-cp313-win_amd64.whl
```

Or download the wheel from the Releases page and `pip install ./<file>.whl`.
Browse the full asset list at
<https://github.com/sparsexlab/torch-amgx/releases/latest>.

Prebuilt wheels are published (24 total = 4 py × 2 os × 3 cuda):

* Linux x86_64, Python 3.10 / 3.11 / 3.12 / 3.13, CUDA 12.4 / 12.6 / 12.8
* Windows x64, Python 3.10 / 3.11 / 3.12 / 3.13, CUDA 12.4 / 12.6 / 12.8

> AmgX is **CUDA-only**, so there are no macOS or AMD/ROCm wheels.

The AmgX shared library is bundled into the wheel (`auditwheel` /
`delvewheel`) so no separate AmgX SDK install is required. You still need
a matching NVIDIA driver and a `torch` build for the same CUDA series.

## Quick start

```python
import torch
import torch_amgx

# Build a sparse SPD system on CUDA
A_csr_indptr, A_csr_indices, A_csr_values = ...  # all torch cuda tensors
shape = (n, n)
b = torch.randn(n, device="cuda", dtype=torch.float64)

# One-shot forward solve (no autograd -- see torch-sla for the autograd-wrapped variant)
x = torch_amgx.solve_csr(A_csr_indptr, A_csr_indices, A_csr_values, shape, b)

# Reusable solver (one setup, many RHS)
cfg = torch_amgx.Config(method="pbicgstab", tol=1e-8, maxiter=200)
solver = torch_amgx.Solver(cfg)
solver.setup_csr(A_csr_indptr, A_csr_indices, A_csr_values, shape)
for b in rhs_stream:
    x = solver.solve(b)              # warm; same matrix, new RHS
```

## Methods

| `method=` | Underlying AmgX config |
|---|---|
| `"pbicgstab"` (default) | PBiCGStab + Classical AMG V-cycle |
| `"pcg"` | PCG + AMG |
| `"fgmres"` | FGMRES + AMG |
| `"amg"` | Standalone AMG V-cycle iteration |

For raw printf-style AmgX config string control, pass `Config(amgx_config_str="...")`.

## Relationship to torch-sla

[`torch-sla`](https://github.com/sparsexlab/torch-sla) consumes `torch-amgx` to provide `backend="amgx"` in its unified `solve()` API:

```python
from torch_sla import solve
x = solve(A, b, backend="amgx")     # routes through torch_amgx
```

`torch-sla` wraps `torch_amgx.solve_csr` in its own `torch.autograd.Function`, providing forward + backward (adjoint solve via the conjugate transpose). This keeps the binding here small (just tensors in / out) and the autograd math in one place.

## Build from source

```bash
git clone --recursive https://github.com/sparsexlab/torch-amgx.git
cd torch-amgx
pip install -e .
```

Requires CUDA Toolkit 12.x, CMake 3.18+, and a C++17 compiler (MSVC 2022 on Windows, gcc ≥ 11 on Linux). The `third_party/AMGX` submodule pins the upstream NVIDIA/AMGX commit.

## License

BSD-3-Clause. Bundles NVIDIA AmgX (also BSD-3-Clause).

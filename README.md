# torch-amgx

A **PyTorch-native** binding for [NVIDIA AmgX](https://github.com/NVIDIA/AMGX) — a GPU-resident sparse linear solver library providing algebraic multigrid (AMG) and Krylov methods (PCG, PBiCGStab, FGMRES, ...).

This package is **not a fork of `pyamgx`**. `pyamgx` is a numpy-shaped Cython wrapper that round-trips every tensor through host memory + scipy. `torch-amgx` is built from scratch as a PyTorch C++ extension:

* **Zero-copy**: torch CUDA tensors are passed directly to AmgX via `data_ptr()`. No numpy, no scipy.
* **Autograd-native**: `torch_amgx.solve(A, b)` registers as a torch op with backward (adjoint solve via conjugate-transpose).
* **CUDA stream-aware**: respects `torch.cuda.current_stream()`.
* **Type-safe config**: `Config` is a frozen dataclass, not a printf-style string.
* **Clean lifecycle**: RAII via pybind11; no `STATUS_STACK_BUFFER_OVERRUN` at shutdown.

## Install

```bash
pip install torch-amgx
```

Prebuilt wheels are published for:

* Linux x86_64, Python 3.10 / 3.11 / 3.12, CUDA 12.4 / 12.8
* Windows x86_64, Python 3.10 / 3.11 / 3.12, CUDA 12.4 / 12.8

The AmgX shared library is bundled into the wheel (`auditwheel` / `delvewheel`) so no separate SDK install is required.

## Quick start

```python
import torch
import torch_amgx

# Build a sparse SPD system on CUDA
A_csr_indptr, A_csr_indices, A_csr_values = ...  # all torch cuda tensors
shape = (n, n)
b = torch.randn(n, device="cuda", dtype=torch.float64)

# One-shot solve (autograd-aware)
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

## Build from source

```bash
git clone --recursive https://github.com/sparsexlab/torch-amgx.git
cd torch-amgx
pip install -e .
```

Requires CUDA Toolkit 12.x, CMake 3.18+, and a C++17 compiler (MSVC 2022 on Windows, gcc ≥ 11 on Linux). The `third_party/AMGX` submodule pins the upstream NVIDIA/AMGX commit.

## License

BSD-3-Clause. Bundles NVIDIA AmgX (also BSD-3-Clause).

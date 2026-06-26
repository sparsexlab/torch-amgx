torch-amgx
==========

**torch-amgx** is a PyTorch-native binding for
`NVIDIA AmgX <https://github.com/NVIDIA/AMGX>`_ -- a GPU-resident sparse
linear solver library exposing algebraic multigrid (AMG) and Krylov methods
(PCG, PBiCGStab, FGMRES, ...) as a **torch-native backend**.

Unlike numpy-shaped wrappers, torch-amgx is built from scratch as a PyTorch
C++ extension:

* **Zero-copy** -- torch CUDA tensors are handed to AmgX via ``data_ptr()``;
  no numpy, no scipy round-trip.
* **CUDA-only** -- AmgX runs on NVIDIA GPUs only (CUDA 12.x), including
  Blackwell ``sm_120``. There are no macOS or AMD/ROCm builds.
* **Binding-only** -- this package is the thin tensor-in / tensor-out layer.
  Autograd (adjoint solve via the conjugate transpose) lives in callers like
  `torch-sla <https://github.com/sparsexlab/torch-sla>`_, keeping this repo
  minimal and reusable.
* **CUDA stream-aware** -- respects ``torch.cuda.current_stream()``.
* **Type-safe config** -- :class:`~torch_amgx.Config` is a frozen dataclass,
  not a printf-style string (though a raw config string is still accepted).

Quickstart
----------

.. code-block:: python

   import torch
   import torch_amgx

   # Build a sparse SPD system on CUDA (all tensors live on the GPU).
   n = 1000
   A = torch.randn(n, n, device="cuda", dtype=torch.float64)
   A = A @ A.T + n * torch.eye(n, device="cuda", dtype=torch.float64)  # SPD
   A_csr = A.to_sparse_csr()
   b = torch.randn(n, device="cuda", dtype=torch.float64)

   # One-shot forward solve (no autograd -- see torch-sla for the
   # autograd-wrapped variant).
   x = torch_amgx.solve_csr(
       A_csr.crow_indices(),
       A_csr.col_indices(),
       A_csr.values(),
       (n, n),
       b,
   )

   # Verify the extension is importable and CUDA is available.
   assert torch_amgx.is_available()

For repeated solves on the *same* matrix, build a reusable
:class:`~torch_amgx.Solver` once and call :meth:`~torch_amgx.Solver.solve`
per right-hand side -- see :doc:`usage`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api

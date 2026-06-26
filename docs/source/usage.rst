Usage
=====

All tensors passed to torch-amgx must live on a CUDA device. The matrix must
be **square**; AmgX targets square (typically SPD or structurally symmetric)
systems.

One-shot functional solves
--------------------------

:func:`~torch_amgx.solve_csr` and :func:`~torch_amgx.solve_coo` build a
transient :class:`~torch_amgx.Solver`, run AmgX setup, solve, and tear down.
They are the simplest entry point but pay the setup cost on **every** call.

CSR input
^^^^^^^^^

.. code-block:: python

   import torch
   import torch_amgx

   n = 1000
   A = torch.randn(n, n, device="cuda", dtype=torch.float64)
   A = A @ A.T + n * torch.eye(n, device="cuda", dtype=torch.float64)  # SPD
   A_csr = A.to_sparse_csr()
   b = torch.randn(n, device="cuda", dtype=torch.float64)

   x = torch_amgx.solve_csr(
       A_csr.crow_indices(),
       A_csr.col_indices(),
       A_csr.values(),
       (n, n),
       b,
   )

COO input
^^^^^^^^^

:func:`~torch_amgx.solve_coo` takes ``(values, row, col)`` and converts to CSR
on-device internally:

.. code-block:: python

   coo = A.to_sparse_coo().coalesce()
   row, col = coo.indices()
   x = torch_amgx.solve_coo(coo.values(), row, col, (n, n), b)

Diagnostics
^^^^^^^^^^^

Pass ``return_info=True`` to get a :class:`~torch_amgx.SolveInfo` alongside
the solution:

.. code-block:: python

   x, info = torch_amgx.solve_csr(
       A_csr.crow_indices(), A_csr.col_indices(), A_csr.values(),
       (n, n), b, return_info=True,
   )
   print(info.iter_count, info.residual, info.converged)

Reusable solver (one setup, many RHS)
-------------------------------------

When you solve the *same* matrix against many right-hand sides, build a
:class:`~torch_amgx.Solver` once. :meth:`~torch_amgx.Solver.setup_csr` runs
the (expensive) AmgX setup phase; each :meth:`~torch_amgx.Solver.solve` reuses
it.

.. code-block:: python

   import torch_amgx

   cfg = torch_amgx.Config(method="pbicgstab", tol=1e-8, maxiter=200)
   solver = torch_amgx.Solver(cfg)                       # defaults to cuda:0
   solver.setup_csr(
       A_csr.crow_indices(), A_csr.col_indices(), A_csr.values(), n,
   )

   for b in rhs_stream:                                  # same matrix, new RHS
       x = solver.solve(b)

   # In-place / warm-start variant avoids an output allocation:
   x = torch.empty_like(b)
   solver.solve_into(b, x)

Configuration
-------------

:class:`~torch_amgx.Config` is a frozen dataclass. The ``method`` selects the
outer solver and ``preconditioner`` selects the inner preconditioner:

.. code-block:: python

   torch_amgx.Config(method="pbicgstab", tol=1e-8, maxiter=200)   # default-ish
   torch_amgx.Config(method="pcg", preconditioner="multicolor_dilu")
   torch_amgx.Config(method="amg")                                # standalone AMG

==================  ====================================
``method=``         Underlying AmgX configuration
==================  ====================================
``"pbicgstab"``     PBiCGStab + Classical AMG V-cycle (default)
``"pcg"``           PCG + AMG
``"fgmres"``        FGMRES + AMG
``"amg"``           Standalone AMG V-cycle iteration
==================  ====================================

For full printf-style AmgX control, pass a literal config string, which
short-circuits the method-based construction:

.. code-block:: python

   torch_amgx.Config(amgx_config_str="config_version=2,solver(main)=AMG,...")

Using it through torch-sla
--------------------------

`torch-sla <https://github.com/sparsexlab/torch-sla>`_ consumes torch-amgx to
provide ``backend="amgx"`` in its unified, autograd-aware ``spsolve`` / ``solve``
API. torch-sla wraps :func:`~torch_amgx.solve_csr` in its own
``torch.autograd.Function`` (forward + backward via the adjoint / conjugate
transpose), so the binding here stays small and the autograd math lives in one
place.

.. code-block:: python

   from torch_sla import spsolve

   x = spsolve(A, b, backend="amgx")     # differentiable; routes through torch_amgx

API reference
=============

The public API is re-exported from the top-level :mod:`torch_amgx` package.

.. currentmodule:: torch_amgx

Configuration
-------------

.. autoclass:: torch_amgx.Config
   :members:
   :undoc-members:

.. autoclass:: torch_amgx.SolveInfo
   :members:
   :undoc-members:

Solver
------

.. autoclass:: torch_amgx.Solver
   :members:
   :undoc-members:

Functional solves
-----------------

.. autofunction:: torch_amgx.solve_csr

.. autofunction:: torch_amgx.solve_coo

Introspection
-------------

.. autofunction:: torch_amgx.is_available

.. autofunction:: torch_amgx.amgx_version

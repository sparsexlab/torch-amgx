Installation
============

torch-amgx is **not on PyPI**. Prebuilt wheels are published to
`GitHub Releases <https://github.com/sparsexlab/torch-amgx/releases>`_.

Wheel matrix
------------

There are **24 wheels** per release (4 Python versions x 2 operating systems
x 3 CUDA toolkits):

* **Linux** x86_64 -- Python 3.10 / 3.11 / 3.12 / 3.13
* **Windows** x64 -- Python 3.10 / 3.11 / 3.12 / 3.13
* CUDA **12.4 / 12.6 / 12.8** (the 12.8 build includes Blackwell ``sm_120``)

The AmgX shared library is bundled into the wheel (via ``auditwheel`` on Linux
and ``delvewheel`` on Windows), so no separate AmgX SDK install is required.

.. note::

   AmgX is **CUDA-only**. There are no macOS or AMD/ROCm wheels.

Per-CUDA build tag
------------------

The CUDA toolkit is encoded in the wheel's **build tag** -- ``0_cu124`` /
``0_cu126`` / ``0_cu128`` -- which appears right after the version, e.g.::

   torch_amgx-0.1.0a11-0_cu126-cp311-cp311-manylinux_2_35_x86_64.whl
                       ^^^^^^^

Pick the wheel whose CUDA matches your installed torch. Check it with::

   python -c "import torch; print(torch.version.cuda)"

Map ``12.4 -> 0_cu124``, ``12.6 -> 0_cu126``, ``12.8 -> 0_cu128``.

.. warning::

   **torch-version ABI caveat.** Each wheel is compiled against a *specific*
   torch version's C++ ABI. Installing a wheel that does not match your
   installed torch fails at **import** time -- not install time -- with a
   torch ABI error:

   * Windows: ``ImportError: DLL load failed while importing _C: The
     specified procedure could not be found.``
   * Linux: an ``undefined symbol`` error.

   The fix is to install the matching torch build (same CUDA series *and* a
   compatible torch version), then reinstall the corresponding torch-amgx
   wheel.

Installing
----------

Install the release asset URL directly with ``pip install --no-deps`` (so pip
doesn't try to resolve/upgrade your carefully-pinned torch):

.. code-block:: bash

   # Linux x86_64, Python 3.11, torch built for CUDA 12.6 -> 0_cu126
   pip install --no-deps \
     https://github.com/sparsexlab/torch-amgx/releases/download/v0.1.0a11/torch_amgx-0.1.0a11-0_cu126-cp311-cp311-manylinux_2_35_x86_64.whl

.. code-block:: bash

   # Windows x64, Python 3.13, torch built for CUDA 12.4 -> 0_cu124
   pip install --no-deps ^
     https://github.com/sparsexlab/torch-amgx/releases/download/v0.1.0a11/torch_amgx-0.1.0a11-0_cu124-cp313-cp313-win_amd64.whl

Or download the ``.whl`` from the Releases page and ``pip install ./<file>.whl``.
Browse the full asset list at
https://github.com/sparsexlab/torch-amgx/releases/latest.

You still need a matching NVIDIA driver and a ``torch`` build for the same
CUDA series.

Verifying the install
---------------------

After installing, confirm the native extension loads and CUDA is visible:

.. code-block:: python

   import torch_amgx
   assert torch_amgx.is_available()        # True iff _C imports AND CUDA is up
   print(torch_amgx.amgx_version())        # e.g. "2.5"

:func:`~torch_amgx.is_available` returns ``False`` (rather than raising) when
the extension can't be loaded or CUDA isn't available, so it is safe to gate
on in library code.

Building from source
--------------------

Building requires a GPU toolchain and is only needed if no prebuilt wheel
fits your setup:

.. code-block:: bash

   git clone --recursive https://github.com/sparsexlab/torch-amgx.git
   cd torch-amgx
   pip install -e .

Requires CUDA Toolkit 12.x, CMake 3.18+, and a C++17 compiler (MSVC 2022 on
Windows, gcc >= 11 on Linux). The ``third_party/AMGX`` submodule pins the
upstream NVIDIA/AMGX commit.

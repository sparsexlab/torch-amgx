# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Put the repo root on sys.path so autodoc can import the *pure-Python*
# ``torch_amgx`` package directly from source -- the package is NOT installed
# on the (GPU-less) Read the Docs builder. The compiled CUDA extension
# (``torch_amgx._C``) is mocked below; only it requires CUDA + AmgX.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information ------------------------------------------------------
project = "torch-amgx"
copyright = "2026, sparsexlab"
author = "sparsexlab"

try:
    from torch_amgx import __version__ as release
except Exception:  # pragma: no cover - autodoc still works without it
    release = "0.1.0"
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autodoc ------------------------------------------------------------------
# ``torch_amgx._C`` is the compiled CUDA/AmgX extension; it cannot be imported
# on the RTD builder (no GPU, not built). Mock ONLY the native extension so the
# rest of the pure-Python package imports and documents normally. Real
# torch-cpu is installed so torch types resolve.
autodoc_mock_imports = ["torch_amgx._C"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "torch-amgx"

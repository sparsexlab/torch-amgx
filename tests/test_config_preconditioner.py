"""Unit tests for the new preconditioner field on Config.

These tests are pure ``build_config_str()`` checks -- they do **not**
require CUDA / AmgX runtime, so they run on every developer machine.
"""
from __future__ import annotations

import pytest

from torch_amgx import Config


# ---------------------------------------------------------------- defaults
def test_default_preconditioner_is_amg():
    cfg = Config(method="pcg")
    s = cfg.build_config_str()
    assert "preconditioner(amg)=AMG" in s, s
    assert "amg:cycle=V" in s, s


def test_standalone_amg_ignores_preconditioner_field():
    """Method == 'amg' means AMG IS the solver; the preconditioner field
    is meaningless and should be dropped silently."""
    cfg = Config(method="amg", preconditioner="multicolor_dilu")
    s = cfg.build_config_str()
    assert "solver(main)=AMG" in s, s
    assert "preconditioner" not in s, s


# ---------------------------------------------------------------- non-AMG
@pytest.mark.parametrize("name, token", [
    ("jacobi_l1",       "JACOBI_L1"),
    ("block_jacobi",    "BLOCK_JACOBI"),
    ("multicolor_gs",   "MULTICOLOR_GS"),
    ("multicolor_dilu", "MULTICOLOR_DILU"),
    ("multicolor_ilu",  "MULTICOLOR_ILU"),
    ("chebyshev",       "CHEBYSHEV_POLY"),
    ("polynomial",      "POLYNOMIAL"),
    ("kaczmarz",        "KACZMARZ"),
    ("none",            "NOSOLVER"),
])
def test_preconditioner_emits_amgx_token(name, token):
    cfg = Config(method="pcg", preconditioner=name)
    s = cfg.build_config_str()
    assert token in s, f"{name!r} -> {s}"
    # The outer solver must still be PCG, not replaced by the preconditioner.
    assert "solver(main)=PCG" in s, s


def test_chebyshev_carries_polynomial_order():
    """Chebyshev's only knob (polynomial_order) is emitted as a sane default."""
    cfg = Config(method="pbicgstab", preconditioner="chebyshev")
    s = cfg.build_config_str()
    assert "chebyshev_polynomial_order" in s, s


# ---------------------------------------------------------------- errors
def test_unknown_preconditioner_raises():
    with pytest.raises(ValueError, match="Unknown preconditioner"):
        Config(method="pcg", preconditioner="not-a-real-name").build_config_str()


# ---------------------------------------------------------------- escape hatch
def test_amgx_config_str_short_circuits_all_fields():
    explicit = "config_version=2,solver(main)=AMG,main:max_iters=7"
    cfg = Config(method="pcg", preconditioner="multicolor_dilu",
                 amgx_config_str=explicit)
    assert cfg.build_config_str() == explicit

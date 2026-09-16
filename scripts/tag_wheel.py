#!/usr/bin/env python3
"""Inject a per-CUDA marker into a wheel's *build tag* so the three CUDA
variants (cu12.4 / cu12.6 / cu12.8) of the same (os, python, version) don't
collide as GitHub Release assets, and so `pip install <url>` lands the right
native build for the user's torch CUDA.

torch-amgx binds NVIDIA AmgX and compiles against torch's C++ ABI for a
specific CUDA toolkit. The (os, python) wheel filename alone is identical for
all three CUDA builds, so without a build tag they overwrite each other on the
Release -- only the last (cu12.8) survives, and installing it onto a host with
e.g. torch 2.6.0+cu124 fails at runtime with a torch C++ ABI mismatch
(ImportError: DLL load failed while importing _C).

The tag also carries the *torch* version, because CUDA alone does not
identify the ABI: the extension links torch's C++ ABI, which torch does not
keep stable across minor releases, so a wheel built against torch 2.11 can
fail to import on torch 2.10 with the same CUDA. The torch version is read
from ``torch_amgx/_build_info.py`` *inside the wheel* (written by setup.py at
build time) rather than passed in, so the tag cannot drift from what was
actually compiled.

A wheel filename is:
    {distribution}-{version}[-{build tag}]-{python}-{abi}-{platform}.whl
PEP 427 requires the build tag to start with a digit, so we use a tag like
    0_cu126_torch211
which sorts/parses cleanly and is visible in the filename, e.g.
    torch_amgx-0.1.0a2-0_cu126_torch211-cp313-cp313-win_amd64.whl

Usage:
    python tag_wheel.py <cuda> <wheel-dir> [<out-dir>]

<cuda> is the matrix CUDA value, with or without the dot ("12.6" or "cu126");
it is normalised to 0_cu<digits>, and the torch part is appended if the wheel
carries build info.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

WHEEL_RE = re.compile(
    r"^(?P<dist>.+?)-(?P<ver>.+?)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$"
)


def cuda_to_build(cuda: str) -> str:
    """Normalise a matrix cuda value ("12.6", "cu126", "12-6") to a PEP 427
    build tag ("0_cu126"). The build tag must start with a digit."""
    digits = re.sub(r"\D", "", cuda)  # strip dots/cu/anything non-numeric
    if not digits:
        raise SystemExit(f"could not derive cuda digits from: {cuda!r}")
    return f"0_cu{digits}"


def torch_tag(whl: Path) -> Optional[str]:
    """Read the torch version the wheel was built against, as "torch211".

    Reads ``torch_amgx/_build_info.py`` from inside the wheel; returns None if
    it is absent, so a wheel built by an older setup.py still tags cleanly on
    CUDA alone. The dot is dropped to match the ``cu128`` convention: torch
    2.11 -> ``torch211``. That is ambiguous only for a minor version >= 100,
    which does not exist.
    """
    with zipfile.ZipFile(whl) as z:
        names = [n for n in z.namelist() if n.endswith("torch_amgx/_build_info.py")]
        if not names:
            return None
        text = z.read(names[0]).decode("utf-8")
    m = re.search(r"^TORCH_VERSION\s*=\s*['\"]([^'\"]+)", text, re.M)
    if not m:
        return None
    # "2.11.0+cu128" -> major, minor
    base = m.group(1).split("+")[0]
    parts = base.split(".")
    if len(parts) < 2:
        return None
    return f"torch{parts[0]}{parts[1]}"


def retag(whl: Path, build: str, out_dir: Path) -> Path:
    m = WHEEL_RE.match(whl.name)
    if not m:
        raise SystemExit(f"unrecognised wheel name: {whl.name}")
    g = m.groupdict()
    tt = torch_tag(whl)
    if tt:
        build = f"{build}_{tt}"
    else:
        print(f"  warning: {whl.name} carries no _build_info.py; "
              f"tagging on CUDA alone, so the filename cannot promise a torch ABI")
    new_name = f"{g['dist']}-{g['ver']}-{build}-{g['py']}-{g['abi']}-{g['plat']}.whl"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / new_name
    shutil.copy2(whl, dst)

    # Update the build tag inside the wheel's WHEEL metadata too, so
    # `pip` and `wheel unpack` stay consistent with the filename.
    _rewrite_build_in_wheel(dst, build)
    print(f"{whl.name}  ->  {dst.name}")
    return dst


def _rewrite_build_in_wheel(whl: Path, build: str) -> None:
    tmp = whl.with_suffix(".whl.tmp")
    with zipfile.ZipFile(whl) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".dist-info/WHEEL"):
                text = data.decode("utf-8")
                if "Build:" in text:
                    text = re.sub(r"(?m)^Build:.*$", f"Build: {build}", text)
                else:
                    text = text.rstrip("\n") + f"\nBuild: {build}\n"
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(whl)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    build = cuda_to_build(argv[1])
    wheel_dir = Path(argv[2])
    out_dir = Path(argv[3]) if len(argv) > 3 else wheel_dir
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheel_dir}")
    for whl in wheels:
        retag(whl, build, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

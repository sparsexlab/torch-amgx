#!/usr/bin/env bash
# Build the vendored NVIDIA AmgX shared library + headers into
# ./build/amgx/install. Used by setup.py (via the AMGX_DIR env var) and
# by the GitHub Actions wheel CI.
#
# Requires CUDA Toolkit 12.x, CMake 3.18+, and a C++17 compiler.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AMGX_SRC="${AMGX_SRC:-${REPO_ROOT}/third_party/AMGX}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/amgx}"
CUDA_ARCH="${CUDA_ARCH:-70;80;89;90;120}"

if [[ ! -d "${AMGX_SRC}" ]]; then
    echo "AmgX source not found at ${AMGX_SRC}." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "=== Configuring AmgX (fat binary sm_${CUDA_ARCH//;/, sm_}) ==="
cmake -S "${AMGX_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DAMGX_NO_RPATH=ON

echo "=== Building AmgX (this is the long step, ~30 min) ==="
# Cap parallelism to avoid OOM on memory-constrained boxes (WSL2 caps at
# ~8 GB by default; nvcc + multi-arch fat-binary compilation can easily
# blow past that with unlimited -j). Users with more RAM can override.
PARALLEL="${CMAKE_PARALLEL:-2}"
cmake --build "${BUILD_DIR}" --config Release --parallel "${PARALLEL}"

echo "=== Installing AmgX to ${BUILD_DIR}/install ==="
cmake --install "${BUILD_DIR}" --config Release

echo "=== Done. AMGX_DIR=${BUILD_DIR}/install ==="
ls -la "${BUILD_DIR}/install/lib" || true

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLANG_SRC_DIR="${ROOT_DIR}/slang"
SLANG_BUILD_DIR="${SLANG_SRC_DIR}/__CMake_build_external"

if [[ ! -f "${SLANG_SRC_DIR}/CMakeLists.txt" ]]; then
  echo "missing Slang source at ${SLANG_SRC_DIR}" >&2
  echo "expected layout: ./slang/CMakeLists.txt Have you initialized the git submodules, or cloned with --recursive?" >&2
  exit 1
fi

cmake \
  -S "${SLANG_SRC_DIR}" \
  -B "${SLANG_BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSLANG_LIB_TYPE=STATIC \
  -DSLANG_ENABLE_SLANG_RHI=OFF \
  -DSLANG_ENABLE_GFX=OFF \
  -DSLANG_ENABLE_TESTS=OFF \
  -DSLANG_ENABLE_EXAMPLES=OFF \
  -DSLANG_ENABLE_SLANGD=OFF \
  -DSLANG_ENABLE_SLANGI=OFF \
  -DSLANG_ENABLE_REPLAYER=OFF \
  -DSLANG_ENABLE_CUDA=OFF \
  -DSLANG_ENABLE_OPTIX=OFF \
  -DSLANG_ENABLE_AFTERMATH=OFF \
  -DSLANG_ENABLE_PREBUILT_BINARIES=OFF \
  -DSLANG_SLANG_LLVM_FLAVOR=DISABLE \
  -DSLANG_EMBED_CORE_MODULE=OFF

cmake --build "${SLANG_BUILD_DIR}" --target slang

echo "Built Slang at ${SLANG_BUILD_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"
TARGET_OS="${2:-linux}"
TARGET_ARCH="${3:-x86_64}"
INSTALL_DIR="${4:-${ROOT_DIR}/install}"
OUT_DIR="${5:-${ROOT_DIR}/dist}"

if [[ -z "${VERSION}" ]]; then
  echo "usage: $0 <version> [os] [arch] [install-dir] [out-dir]" >&2
  exit 1
fi

if [[ ! -x "${INSTALL_DIR}/bin/shader_explorer" ]]; then
  echo "missing installed binary at ${INSTALL_DIR}/bin/shader_explorer" >&2
  echo "run: ninja -C build install" >&2
  exit 1
fi

PACKAGE_BASENAME="shader_explorer-${VERSION}-${TARGET_OS}-${TARGET_ARCH}"
PACKAGE_ROOT="${OUT_DIR}/${PACKAGE_BASENAME}"
ARCHIVE_PATH="${OUT_DIR}/${PACKAGE_BASENAME}.tar.gz"

rm -rf "${PACKAGE_ROOT}"
mkdir -p "${PACKAGE_ROOT}"
cp -RP "${INSTALL_DIR}/." "${PACKAGE_ROOT}/"
rm -f "${ARCHIVE_PATH}"

tar \
  --create \
  --gzip \
  --file "${ARCHIVE_PATH}" \
  --directory "${OUT_DIR}" \
  "${PACKAGE_BASENAME}"

printf '%s\n' "${ARCHIVE_PATH}"

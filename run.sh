#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
GTSAM_PREFIX="${ROOT_DIR}/.deps/gtsam-install"

if [[ -d "${GTSAM_PREFIX}" ]]; then
  export CMAKE_PREFIX_PATH="${GTSAM_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

exec "${BUILD_DIR}/bin/macos-metal-vslam"

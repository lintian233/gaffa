#!/usr/bin/env bash
set -euo pipefail

readonly prefix=/opt/wheel-deps
readonly cuda_architectures=${CMAKE_CUDA_ARCHITECTURES:-75\;80\;86\;89\;90\;100\;120}
readonly cpm_cache=/opt/cpm-cache
readonly cpm_file=${cpm_cache}/cpm/CPM_0.42.0.cmake
readonly cpm_sha256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a

mkdir -p "${cpm_cache}/cpm"
if ! echo "${cpm_sha256}  ${cpm_file}" | sha256sum --check --strict --status; then
  curl --fail --location --retry 5 \
      --connect-timeout 15 --max-time 300 \
      --output "${cpm_file}" \
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
fi
echo "${cpm_sha256}  ${cpm_file}" | sha256sum --check --strict

export CPM_SOURCE_CACHE="${cpm_cache}"

cmake -S /src/loki -B /build/loki -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/loki \
  -DCMAKE_PREFIX_PATH="${prefix}" \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DENABLE_CUDA=ON \
  -DLOKI_BUILD_SHARED=ON \
  -DLOKI_ENABLE_NATIVE_ARCH=OFF
cmake --build /build/loki --parallel
cmake --install /build/loki

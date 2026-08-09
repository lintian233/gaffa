#!/usr/bin/env bash
set -euo pipefail

readonly prefix=/opt/wheel-deps
readonly sources=/opt/wheel-sources
readonly builds=/opt/wheel-build

mkdir -p "${prefix}" "${sources}" "${builds}"

fetch() {
  local name=$1
  local url=$2
  local checksum=$3
  local archive="${sources}/${name}.tar.gz"

  curl --fail --location --retry 5 \
      --connect-timeout 15 --max-time 300 \
      --output "${archive}" "${url}"
  echo "${checksum}  ${archive}" | sha256sum --check --strict
  mkdir -p "${sources}/${name}"
  tar --extract --gzip --file "${archive}" \
      --directory "${sources}/${name}" --strip-components=1
}

fetch fmt \
  https://codeload.github.com/fmtlib/fmt/tar.gz/refs/tags/12.0.0 \
  aa3e8fbb6a0066c03454434add1f1fc23299e85758ceec0d7d2d974431481e40
fetch fftw \
  https://deb.debian.org/debian/pool/main/f/fftw3/fftw3_3.3.10.orig.tar.gz \
  56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
fetch hdf5 \
  https://codeload.github.com/HDFGroup/hdf5/tar.gz/refs/tags/hdf5_1.14.6 \
  09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f
fetch yaml-cpp \
  https://codeload.github.com/jbeder/yaml-cpp/tar.gz/refs/tags/0.8.0 \
  fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16

cmake -S "${sources}/fmt" -B "${builds}/fmt" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DBUILD_SHARED_LIBS=ON \
  -DFMT_DOC=OFF \
  -DFMT_TEST=OFF
cmake --build "${builds}/fmt" --parallel
cmake --install "${builds}/fmt"

cmake -S "${sources}/hdf5" -B "${builds}/hdf5" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_STATIC_LIBS=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_JAVA=OFF \
  -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_ENABLE_ROS3_VFD=OFF
cmake --build "${builds}/hdf5" --parallel
cmake --install "${builds}/hdf5"

(
  cd "${sources}/fftw"
  ./configure \
    --prefix="${prefix}" \
    --enable-float \
    --enable-openmp \
    --enable-shared \
    --disable-static \
    --disable-doc
  make --jobs="$(nproc)"
  make install
)

cmake -S "${sources}/yaml-cpp" -B "${builds}/yaml-cpp" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DYAML_BUILD_SHARED_LIBS=OFF \
  -DYAML_CPP_BUILD_CONTRIB=OFF \
  -DYAML_CPP_BUILD_TESTS=OFF \
  -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build "${builds}/yaml-cpp" --parallel
cmake --install "${builds}/yaml-cpp"

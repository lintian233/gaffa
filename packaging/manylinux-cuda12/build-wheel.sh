#!/usr/bin/env bash
set -euo pipefail

readonly python=/opt/python/cp312-cp312/bin/python
readonly prefix=/opt/wheel-deps
readonly cuda_architectures=${CMAKE_CUDA_ARCHITECTURES:-75\;80\;86\;89\;90\;100\;120}
readonly pybind11_dir=$("${python}" -m pybind11 --cmakedir)

mkdir -p /src/gaffa/python/gaffa/licenses
cp /opt/wheel-sources/fmt/LICENSE \
   /src/gaffa/python/gaffa/licenses/FMT_LICENSE
cp /opt/wheel-sources/fftw/COPYING \
   /src/gaffa/python/gaffa/licenses/FFTW_LICENSE
cp /opt/wheel-sources/hdf5/COPYING \
   /src/gaffa/python/gaffa/licenses/HDF5_LICENSE
cp /opt/wheel-sources/yaml-cpp/LICENSE \
   /src/gaffa/python/gaffa/licenses/YAML_CPP_LICENSE

rm -rf /dist /wheelhouse /build/gaffa-wheel
mkdir -p /dist /wheelhouse

"${python}" -m build --wheel --no-isolation --outdir /dist \
  -Cbuild-dir=/build/gaffa-wheel \
  -Ccmake.args="-DCMAKE_BUILD_TYPE=Release" \
  -Ccmake.args="-DBUILD_TESTING=OFF" \
  -Ccmake.args="-DGAFFA_ENABLE_LOKI=ON" \
  -Ccmake.args="-DGAFFA_BUNDLE_LOKI=ON" \
  -Ccmake.args="-DGAFFA_LOKI_LICENSE_FILE=/src/loki/LICENSE" \
  -Ccmake.args="-DCMAKE_PREFIX_PATH=${prefix};/opt/loki;${pybind11_dir}" \
  -Ccmake.args="-DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures}"

raw_wheels=(/dist/*.whl)
if (( ${#raw_wheels[@]} != 1 )); then
  echo "expected exactly one raw wheel, found ${#raw_wheels[@]}" >&2
  exit 1
fi

auditwheel repair \
  --plat manylinux_2_28_x86_64 \
  --exclude libcuda.so.1 \
  --exclude libcudart.so.12 \
  --exclude libcufft.so.11 \
  --exclude libcurand.so.10 \
  --wheel-dir /wheelhouse \
  "${raw_wheels[0]}"

repaired_wheels=(/wheelhouse/*.whl)
if (( ${#repaired_wheels[@]} != 1 )); then
  echo "expected exactly one repaired wheel, found ${#repaired_wheels[@]}" >&2
  exit 1
fi

auditwheel show "${repaired_wheels[0]}"
"${python}" /src/gaffa/packaging/manylinux-cuda12/verify-wheel.py \
  "${repaired_wheels[0]}"

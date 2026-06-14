#!/usr/bin/env bash

set -e

EXPECTED_CONDA_ENV="gaffa"

if [ -z "$CONDA_PREFIX" ]; then
  echo "Error: CONDA_PREFIX is empty. Please run:"
  echo "  conda activate $EXPECTED_CONDA_ENV"
  return 1 2>/dev/null || exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]; then
  echo "Error: expected conda environment '$EXPECTED_CONDA_ENV', got '${CONDA_DEFAULT_ENV:-unknown}'. Please run:"
  echo "  conda activate $EXPECTED_CONDA_ENV"
  return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CMAKE_CUDA_COMPILER="$CUDACXX"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_CUDA_HOST_COMPILER="$CXX"
export NVCC_CCBIN="$CXX"
export CONAN_HOME="$PWD/.conan2"

# Avoid leaking system CUDA headers/libs into conda CUDA builds.
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDACXX=$CUDACXX"
echo "CC=$CC"
echo "CXX=$CXX"
echo "CMAKE_CUDA_HOST_COMPILER=$CMAKE_CUDA_HOST_COMPILER"
echo "CONAN_HOME=$CONAN_HOME"
echo

which python
python -m pip --version

echo
which nvcc
nvcc --version

echo
"$CC" --version | head -n 1
"$CXX" --version | head -n 1

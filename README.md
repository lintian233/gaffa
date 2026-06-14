# gaffa

Minimal CUDA/C++ and Python hybrid scaffold.

## Environment

```bash
conda env create -f environment.yml
conda activate gaffa
source env/dev.sh
```

The system layer should provide only the NVIDIA driver. The conda environment
provides Python 3.12, CUDA Toolkit 12.8, GCC/G++ 13, CMake, Ninja, and Conan.

## Development

```bash
just install
just test
```

`just install` is the local development shortcut. It first runs Conan using
`conanfile.py` to
generate `build/conan/debug/conan_toolchain.cmake`, then runs editable install
through scikit-build-core with that toolchain.

The equivalent expanded commands are:

```bash
conan install . --build=missing -of build/conan/debug -s build_type=Debug -s compiler.cppstd=20
python -m pip install -e ".[dev]" --no-build-isolation \
  -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake" \
  -Ccmake.args="-DCMAKE_BUILD_TYPE=Debug"
```

For a CMake-only compile check:

```bash
just configure
just build
```

C++ tests use GTest and CTest:

```bash
just test-cpp
```

Run the normal local test suite:

```bash
just test-all
```

Host C++ coverage can be collected from a coverage-instrumented CMake build:

```bash
just coverage-cpp
```

That enables GCC/Clang `--coverage` for host C++ core logic. CUDA kernels are
validated with GPU correctness tests and NVIDIA tools rather than gcov line
coverage:

```bash
just test-cuda
```

`test-cuda` runs Compute Sanitizer `memcheck`, `racecheck`, `initcheck`, and
`synccheck`. It requires a visible NVIDIA GPU.

For a release wheel through scikit-build-core:

```bash
python -m build --wheel --no-isolation
```

`just wheel` wraps the release Conan install and passes the release toolchain to
scikit-build-core.

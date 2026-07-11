set shell := ["bash", "-cu"]

export-env:
    conda env export --from-history > environment.lock.yml

conan-profile:
    source env/dev.sh && { test -f "$CONAN_HOME/profiles/default" || conan profile detect --force; }

deps: conan-profile
    source env/dev.sh && conan install . --build=missing -of build/conan/debug -s build_type=Debug -s compiler.cppstd=20

deps-release: conan-profile
    source env/dev.sh && conan install . --build=missing -of build/conan/release -s build_type=Release -s compiler.cppstd=20

install: deps
    source env/dev.sh && python -m pip install -e ".[dev]" --no-build-isolation --force-reinstall \
      -Cbuild-dir="build/editable-debug" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/debug/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/debug" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/debug" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Debug"

test:
    source env/dev.sh && python -m pytest -v --cov

wheel: deps-release
    source env/dev.sh && python -m build --wheel --no-isolation \
      -Cbuild-dir="build/wheel-release" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/release/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/release" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/release" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Release"

configure: deps
    source env/dev.sh && if [ -f build/dev/CMakeCache.txt ]; then cmake -S . -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Debug; else cmake -S . -B build/dev -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug; fi

reconfigure: deps
    source env/dev.sh && cmake --fresh -S . -B build/dev -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug

build: configure
    source env/dev.sh && cmake --build build/dev

configure-release: deps-release
    source env/dev.sh && if [ -f build/release/CMakeCache.txt ]; then cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release; else cmake -S . -B build/release -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release; fi

build-release: configure-release
    source env/dev.sh && cmake --build build/release

build-benchmarks: configure-release
    source env/dev.sh && cmake --build build/release --target gaffa_benchmarks

bench: build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_filterbank_read_benchmark all tests/data/basedata_240M.fil 1
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_filterbank_read_benchmark all tests/data/Mercer_5_tracking-M03_filtool_01.fil 1

bench-cuda: build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_dedispersion_cuda_benchmark all tests/data/basedata_240M.fil 3
    # source env/dev.sh && /usr/bin/time -v build/release/gaffa_dedispersion_cuda_benchmark all tests/data/FRB20241112A_01.fil 3

bench-spectrum-cpu file="tests/data/Mercer_5_tracking-M03_filtool_01.fil" dm="1050.2" iterations="3" chan_begin="0" chan_end="0": build-benchmarks
    source env/dev.sh && /usr/bin/time -v build/release/gaffa_dedispersion_spectrum_cpu_benchmark {{file}} {{dm}} {{iterations}} {{chan_begin}} {{chan_end}}

bench-dm-search file="tests/data/19C113_J1846-05N2_onoff-M02_add_0001-0466_filtool_01.fil" backend="cuda-subband" ndm="256" dm_low="150" dm_step="0.5" period_min="0.018" period_max="1" max_peaks="0" print_peaks="64" snr_threshold="7.5" bins_min="180" bins_max="256" preprocess="riptide" running_median_seconds="5" subband_channels="32" ndm_per_nominal="32" max_candidates="0" omp_threads="56": build-benchmarks
    source env/dev.sh && OMP_NUM_THREADS={{omp_threads}} OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false /usr/bin/time -v build/release/gaffa_dm_search_benchmark {{file}} {{backend}} {{ndm}} {{dm_low}} {{dm_step}} {{period_min}} {{period_max}} {{max_peaks}} {{print_peaks}} {{snr_threshold}} {{bins_min}} {{bins_max}} {{preprocess}} {{running_median_seconds}} {{subband_channels}} {{ndm_per_nominal}} {{max_candidates}}

bench-ffa-transform mode="all" rows="73242" bins="256" nseries="32" iterations="5" warmup="1" validate="no-validate" omp_threads="56": build-benchmarks
    source env/dev.sh && OMP_NUM_THREADS={{omp_threads}} OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false /usr/bin/time -v build/release/gaffa_ffa_transform_benchmark {{mode}} {{rows}} {{bins}} {{nseries}} {{iterations}} {{warmup}} {{validate}}

test-cpp: build
    source env/dev.sh && ctest --test-dir build/dev --output-on-failure

test-cuda: build
    source env/dev.sh && compute-sanitizer --tool memcheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool racecheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool initcheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool synccheck --error-exitcode 1 build/dev/gaffa_cpp_tests

coverage-cpp: deps
    source env/dev.sh && mkdir -p coverage && if [ -f build/coverage/CMakeCache.txt ]; then cmake -S . -B build/coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; else cmake -S . -B build/coverage -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; fi && cmake --build build/coverage --target clean && { find build/coverage -name '*.gcda' -delete -o -name '*.gcno' -delete 2>/dev/null || true; } && cmake --build build/coverage && ctest --test-dir build/coverage --output-on-failure && find build/coverage -name 'cmake_device_link.gcno' -delete -o -name 'cmake_device_link.gcda' -delete -o -name 'link.stub*.gcno' -delete -o -name 'link.stub*.gcda' -delete && "$CONDA_PREFIX/bin/gcovr" --gcov-executable "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcov" --root . --filter src/gaffa --filter include/gaffa --exclude tests/cpp --exclude src/gaffa/bindings.cpp --exclude 'src/gaffa/python/.*' --exclude src/gaffa/io/filterbank_legacy.cpp --exclude '.*cmake_device_link.*' --exclude '.*link\.stub.*' --txt --xml-pretty --xml coverage/cpp.xml --html-details coverage/cpp.html

test-all: test test-cpp test-cuda

clean:
    rm -rf build dist *.egg-info

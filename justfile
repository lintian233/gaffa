set shell := ["bash", "-cu"]

# Shared paths and environment setup.
debug_dir := "build/dev"
release_dir := "build/release"
loki_dir := "build/loki"
loki_release_dir := "build/loki-release"
dev_env := "source env/dev.sh &&"

# Environment and dependencies

export-env:
    conda env export --from-history > environment.lock.yml

conan-profile:
    {{ dev_env }} { test -f "$CONAN_HOME/profiles/default" || conan profile detect --force; }

deps-debug: conan-profile
    {{ dev_env }} conan install . --build=missing -of build/conan/debug -s build_type=Debug -s compiler.cppstd=20

deps-release: conan-profile
    {{ dev_env }} conan install . --build=missing -of build/conan/release -s build_type=Release -s compiler.cppstd=20

# Backward-compatible name for the default development dependency set.
deps: deps-debug

# Configure and build

configure: deps-debug
    {{ dev_env }} if [ -f {{ debug_dir }}/CMakeCache.txt ]; then cmake -S . -B {{ debug_dir }} -G Ninja -DCMAKE_BUILD_TYPE=Debug; else cmake -S . -B {{ debug_dir }} -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug; fi

reconfigure: deps-debug
    {{ dev_env }} cmake --fresh -S . -B {{ debug_dir }} -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug

build: configure
    {{ dev_env }} cmake --build {{ debug_dir }}

configure-release: deps-release
    {{ dev_env }} if [ -f {{ release_dir }}/CMakeCache.txt ]; then cmake -S . -B {{ release_dir }} -G Ninja -DCMAKE_BUILD_TYPE=Release; else cmake -S . -B {{ release_dir }} -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release; fi

build-release: configure-release
    {{ dev_env }} cmake --build {{ release_dir }}

build-benchmarks: configure-release
    {{ dev_env }} cmake --build {{ release_dir }} --target gaffa_benchmarks

install: deps-debug
    {{ dev_env }} python -m pip install -e ".[dev]" --no-build-isolation --force-reinstall \
      -Cbuild-dir="build/editable-debug" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/debug/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/debug" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/debug" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Debug"

wheel: deps-release
    {{ dev_env }} python -m build --wheel --no-isolation \
      -Cbuild-dir="build/wheel-release" \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=$PWD/build/conan/release/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_PREFIX_PATH=$PWD/build/conan/release" \
      -Ccmake.args="-Dpybind11_DIR=$PWD/build/conan/release" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Release"

# Optional Loki integration. CMake discovers Loki through standard package
# paths, LOKI_ROOT, or the conventional $HOME/opt/loki prefix.

configure-loki: deps-debug
    {{ dev_env }} if [ -f {{ loki_dir }}/CMakeCache.txt ]; then cmake -S . -B {{ loki_dir }} -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_LOKI=ON; else cmake -S . -B {{ loki_dir }} -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_LOKI=ON; fi

build-loki: configure-loki
    {{ dev_env }} cmake --build {{ loki_dir }} --target gaffa_loki

test-loki: configure-loki
    {{ dev_env }} cmake --build {{ loki_dir }} --target gaffa_loki_smoke_tests gaffa_loki_pffa_tests
    {{ dev_env }} ctest --test-dir {{ loki_dir }} -R "^(LokiSmoke|LokiPffaPlan|LokiPffaProgram)\\." --output-on-failure

configure-loki-release: deps-release
    {{ dev_env }} if [ -f {{ loki_release_dir }}/CMakeCache.txt ]; then cmake -S . -B {{ loki_release_dir }} -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAFFA_ENABLE_LOKI=ON; else cmake -S . -B {{ loki_release_dir }} -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DGAFFA_ENABLE_LOKI=ON; fi

# Tests and quality checks

test-python:
    {{ dev_env }} python -m pytest -v --cov

typecheck-python:
    {{ dev_env }} pyright

# Backward-compatible Python test entry point.
test: typecheck-python test-python

test-cpp: build
    {{ dev_env }} ctest --test-dir {{ debug_dir }} --output-on-failure

test-cuda: build
    {{ dev_env }} compute-sanitizer --tool memcheck --error-exitcode 1 {{ debug_dir }}/gaffa_cpp_tests
    {{ dev_env }} compute-sanitizer --tool racecheck --error-exitcode 1 {{ debug_dir }}/gaffa_cpp_tests
    {{ dev_env }} compute-sanitizer --tool initcheck --error-exitcode 1 {{ debug_dir }}/gaffa_cpp_tests
    {{ dev_env }} compute-sanitizer --tool synccheck --error-exitcode 1 {{ debug_dir }}/gaffa_cpp_tests

coverage-cpp: deps-debug
    {{ dev_env }} mkdir -p coverage && if [ -f build/coverage/CMakeCache.txt ]; then cmake -S . -B build/coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; else cmake -S . -B build/coverage -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; fi && cmake --build build/coverage --target clean && { find build/coverage -name '*.gcda' -delete -o -name '*.gcno' -delete 2>/dev/null || true; } && cmake --build build/coverage && ctest --test-dir build/coverage --output-on-failure && find build/coverage -name 'cmake_device_link.gcno' -delete -o -name 'cmake_device_link.gcda' -delete -o -name 'link.stub*.gcno' -delete -o -name 'link.stub*.gcda' -delete && "$CONDA_PREFIX/bin/gcovr" --gcov-executable "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcov" --root . --filter src/gaffa --filter include/gaffa --exclude tests/cpp --exclude src/gaffa/bindings.cpp --exclude 'src/gaffa/python/.*' --exclude src/gaffa/io/filterbank_legacy.cpp --exclude '.*cmake_device_link.*' --exclude '.*link\.stub.*' --txt --xml-pretty --xml coverage/cpp.xml --html-details coverage/cpp.html

test-all: test-python test-cpp test-cuda

# Benchmarks

bench-filterbank: build-benchmarks
    {{ dev_env }} /usr/bin/time -v {{ release_dir }}/gaffa_filterbank_read_benchmark all tests/data/basedata_240M.fil 1
    {{ dev_env }} /usr/bin/time -v {{ release_dir }}/gaffa_filterbank_read_benchmark all tests/data/PT2023_0192_Mercer_5_20231221_c8_t4_filtool_01.fil 1

bench-dedispersion-cuda: build-benchmarks
    {{ dev_env }} /usr/bin/time -v {{ release_dir }}/gaffa_dedispersion_cuda_benchmark all tests/data/basedata_240M.fil 3

bench-spectrum-cpu file="tests/data/PT2023_0192_Mercer_5_20231221_c8_t4_filtool_01.fil" dm="1050.2" iterations="3" chan_begin="0" chan_end="0": build-benchmarks
    {{ dev_env }} /usr/bin/time -v {{ release_dir }}/gaffa_dedispersion_spectrum_cpu_benchmark {{ file }} {{ dm }} {{ iterations }} {{ chan_begin }} {{ chan_end }}

bench-dm-search file="tests/data/PT2023_0192_Mercer_5_20231221_c8_t4_filtool_01.fil" backend="cuda-subband" ndm="64" dm_low="200" dm_step="0.5" period_min="0.018" period_max="1" max_peaks="0" print_peaks="64" snr_threshold="7.5" bins_min="180" bins_max="256" preprocess="riptide" running_median_seconds="5" subband_channels="32" ndm_per_nominal="32" max_candidates="0" omp_threads="56": build-benchmarks
    {{ dev_env }} OMP_NUM_THREADS={{ omp_threads }} OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false /usr/bin/time -v {{ release_dir }}/gaffa_dm_search_benchmark {{ file }} {{ backend }} {{ ndm }} {{ dm_low }} {{ dm_step }} {{ period_min }} {{ period_max }} {{ max_peaks }} {{ print_peaks }} {{ snr_threshold }} {{ bins_min }} {{ bins_max }} {{ preprocess }} {{ running_median_seconds }} {{ subband_channels }} {{ ndm_per_nominal }} {{ max_candidates }}

bench-cuda-dm-search file="tests/data/PT2023_0192_Mercer_5_20231221_c8_t4_filtool_01.fil" ndm="64" dm_low="200" dm_step="0.5" period_min="0.002" period_max="0.02" snr_threshold="7.5" bins_min="20" bins_max="160" running_median_seconds="5" subband_channels="32" ndm_per_nominal="32": build-benchmarks
    {{ dev_env }} /usr/bin/time -v {{ release_dir }}/gaffa_cuda_dm_search_benchmark {{ file }} {{ ndm }} {{ dm_low }} {{ dm_step }} {{ period_min }} {{ period_max }} {{ snr_threshold }} {{ bins_min }} {{ bins_max }} {{ running_median_seconds }} {{ subband_channels }} {{ ndm_per_nominal }}

bench-ffa-transform mode="all" rows="73242" bins="256" nseries="32" iterations="5" warmup="1" validate="no-validate" omp_threads="56": build-benchmarks
    {{ dev_env }} OMP_NUM_THREADS={{ omp_threads }} OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false /usr/bin/time -v {{ release_dir }}/gaffa_ffa_transform_benchmark {{ mode }} {{ rows }} {{ bins }} {{ nseries }} {{ iterations }} {{ warmup }} {{ validate }}

bench-loki-dm-search file="tests/data/PT2023_0192_Mercer_5_20231221_c8_t4_filtool_01.fil" ndm="64" dm_low="1030" dm_step="0.5" period_min="0.018" period_max="1" snr_threshold="7.5" phase_bins_min="180" phase_bins_max="256" duty_cycle_max="0.2" running_median_seconds="5" subband_channels="32" ndm_per_nominal="32" print_peaks="64" window_mode="zero-pad": configure-loki-release
    {{ dev_env }} cmake --build {{ loki_release_dir }} --target gaffa_loki_dm_search_benchmark
    {{ dev_env }} /usr/bin/time -v {{ loki_release_dir }}/gaffa_loki_dm_search_benchmark {{ file }} {{ ndm }} {{ dm_low }} {{ dm_step }} {{ period_min }} {{ period_max }} {{ snr_threshold }} {{ phase_bins_min }} {{ phase_bins_max }} {{ duty_cycle_max }} {{ running_median_seconds }} {{ subband_channels }} {{ ndm_per_nominal }} {{ print_peaks }} {{ window_mode }}

# Cleanup

clean:
    rm -rf build dist *.egg-info

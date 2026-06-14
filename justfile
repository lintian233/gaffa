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
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Debug"

test:
    source env/dev.sh && python -m pytest -v --cov

wheel: deps-release
    source env/dev.sh && python -m build --wheel --no-isolation \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Release"

configure: deps
    source env/dev.sh && if [ -f build/dev/CMakeCache.txt ]; then cmake -S . -B build/dev -G Ninja -DCMAKE_BUILD_TYPE=Debug; else cmake -S . -B build/dev -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug; fi

reconfigure: deps
    source env/dev.sh && cmake --fresh -S . -B build/dev -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug

build: configure
    source env/dev.sh && cmake --build build/dev

test-cpp: build
    source env/dev.sh && ctest --test-dir build/dev --output-on-failure

test-cuda: build
    source env/dev.sh && compute-sanitizer --tool memcheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool racecheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool initcheck --error-exitcode 1 build/dev/gaffa_cpp_tests
    source env/dev.sh && compute-sanitizer --tool synccheck --error-exitcode 1 build/dev/gaffa_cpp_tests

coverage-cpp: deps
    source env/dev.sh && mkdir -p coverage && if [ -f build/coverage/CMakeCache.txt ]; then cmake -S . -B build/coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; else cmake -S . -B build/coverage -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DGAFFA_ENABLE_COVERAGE=ON; fi && cmake --build build/coverage && ctest --test-dir build/coverage --output-on-failure && find build/coverage -name 'cmake_device_link.gcno' -delete -o -name 'cmake_device_link.gcda' -delete -o -name 'link.stub*.gcno' -delete -o -name 'link.stub*.gcda' -delete && gcovr --gcov-executable "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcov" --root . --filter src/gaffa/vector_add.cpp --filter include --exclude tests/cpp --exclude '.*cmake_device_link.*' --exclude '.*link\.stub.*' --txt --xml-pretty --xml coverage/cpp.xml --html-details coverage/cpp.html

test-all: test test-cpp coverage-cpp

clean:
    rm -rf build dist *.egg-info

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
    source env/dev.sh && python -m pytest

wheel: deps-release
    source env/dev.sh && python -m build --wheel --no-isolation \
      -Ccmake.args="-DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake" \
      -Ccmake.args="-DCMAKE_BUILD_TYPE=Release"

configure: deps
    source env/dev.sh && cmake --fresh -S . -B build/dev -G Ninja -DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug

build:
    source env/dev.sh && cmake --build build/dev

clean:
    rm -rf build dist *.egg-info

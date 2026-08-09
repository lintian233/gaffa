cmake_minimum_required(VERSION 3.26)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
endif()

project(gaffa LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GAFFA_ENABLE_COVERAGE "Build host C++ coverage instrumentation" OFF)
option(GAFFA_ENABLE_LOKI
       "Build optional Loki P-FFA backend and integration tests" OFF)

if(GAFFA_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--coverage>)
    add_link_options(--coverage)
  else()
    message(FATAL_ERROR "GAFFA_ENABLE_COVERAGE requires GCC or Clang for host C++ coverage")
  endif()
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)
find_package(yaml-cpp CONFIG REQUIRED)
include(CTest)

# Loki is an optional periodic-search backend. Core never depends on it; the
# optional gaffa_loki target builds on the stable core API.
if(GAFFA_ENABLE_LOKI)
  set(GAFFA_LOKI_HINTS)
  if(DEFINED ENV{LOKI_ROOT} AND NOT "$ENV{LOKI_ROOT}" STREQUAL "")
    list(APPEND GAFFA_LOKI_HINTS "$ENV{LOKI_ROOT}")
  endif()
  if(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    list(APPEND GAFFA_LOKI_HINTS "$ENV{HOME}/opt/loki")
  endif()
  find_package(loki 0.0.3 CONFIG REQUIRED HINTS ${GAFFA_LOKI_HINTS})

  foreach(GAFFA_LOKI_CONFIGURATION RELEASE RELWITHDEBINFO DEBUG MINSIZEREL)
    get_target_property(
      GAFFA_LOKI_LIBRARY_LOCATION loki::loki
      "IMPORTED_LOCATION_${GAFFA_LOKI_CONFIGURATION}"
    )
    if(GAFFA_LOKI_LIBRARY_LOCATION)
      break()
    endif()
  endforeach()
  if(NOT GAFFA_LOKI_LIBRARY_LOCATION)
    get_target_property(GAFFA_LOKI_LIBRARY_LOCATION loki::loki IMPORTED_LOCATION)
  endif()
  if(NOT GAFFA_LOKI_LIBRARY_LOCATION)
    message(FATAL_ERROR "loki::loki does not expose an imported library location")
  endif()
  get_filename_component(
    GAFFA_LOKI_LIBRARY_DIR "${GAFFA_LOKI_LIBRARY_LOCATION}" DIRECTORY
  )
endif()

file(GLOB_RECURSE GAFFA_CORE_SOURCES CONFIGURE_DEPENDS
  src/gaffa/*.cpp
  src/gaffa/*.cu
)
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/bindings\\.cpp$")
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/python/.*\\.cpp$")
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/.*_legacy\\.cpp$")
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/loki/.*")
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/cuda_memory\\.cu$")

file(GLOB_RECURSE GAFFA_PYTHON_BINDING_SOURCES CONFIGURE_DEPENDS
  src/gaffa/python/*.cpp
)
list(FILTER GAFFA_PYTHON_BINDING_SOURCES EXCLUDE REGEX ".*/loki_bindings\\.cpp$")

# CUDA memory is shared infrastructure. Keep it separate from the core object
# collection so device allocation ownership has one reusable implementation.
add_library(gaffa_cuda_memory STATIC src/gaffa/cuda_memory.cu)
target_link_libraries(gaffa_cuda_memory PUBLIC CUDA::cudart)
target_include_directories(gaffa_cuda_memory PUBLIC include)
set_target_properties(gaffa_cuda_memory PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

add_library(gaffa_core_objects OBJECT ${GAFFA_CORE_SOURCES})
target_link_libraries(gaffa_core_objects PUBLIC gaffa_cuda_memory CUDA::cudart CUDA::cuda_driver)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gaffa_core_objects PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(gaffa_core_objects PUBLIC include)
set_target_properties(gaffa_core_objects PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

add_library(gaffa_core STATIC $<TARGET_OBJECTS:gaffa_core_objects>)
target_link_libraries(gaffa_core PUBLIC
  gaffa_cuda_memory
  CUDA::cudart
  CUDA::cuda_driver
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gaffa_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(gaffa_core PUBLIC include)
set_target_properties(gaffa_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

if(GAFFA_ENABLE_LOKI)
  add_library(gaffa_loki STATIC
    src/gaffa/loki/loki_pffa_plan.cpp
    src/gaffa/loki/loki_pffa.cu
    src/gaffa/loki/loki_dm_search.cu
  )
  target_link_libraries(gaffa_loki PUBLIC
    gaffa_core
    loki::loki
  )
  target_include_directories(gaffa_loki PUBLIC include)
  set_target_properties(gaffa_loki PROPERTIES
    BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )
endif()

pybind11_add_module(_core NO_EXTRAS
  src/gaffa/bindings.cpp
  ${GAFFA_PYTHON_BINDING_SOURCES}
)
target_link_libraries(_core PRIVATE gaffa_core)
target_include_directories(_core PRIVATE include)
set_target_properties(_core PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

if(GAFFA_ENABLE_LOKI)
  # Keep the optional Loki Python extension outside _core. Importing gaffa._core
  # must not require Loki or change the dependency graph of the core package.
  pybind11_add_module(_loki NO_EXTRAS
    src/gaffa/python/loki_bindings.cpp
  )
  target_link_libraries(_loki PRIVATE gaffa_loki)
  target_include_directories(_loki PRIVATE include)
  set_target_properties(_loki PROPERTIES
    BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
    INSTALL_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )
endif()

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  file(GLOB_RECURSE GAFFA_CPP_TEST_SOURCES CONFIGURE_DEPENDS
    tests/cpp/test_*.cpp
  )

  add_executable(gaffa_cpp_tests
    src/gaffa/io/filterbank_legacy.cpp
    ${GAFFA_CPP_TEST_SOURCES}
  )
  target_link_libraries(gaffa_cpp_tests PRIVATE GTest::gtest_main gaffa_core)
  target_compile_definitions(gaffa_cpp_tests
    PRIVATE
      GAFFA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
      GAFFA_ENABLE_LEGACY_FILTERBANK_REFERENCE
  )
  target_include_directories(gaffa_cpp_tests PRIVATE include include/gaffa)
  set_target_properties(gaffa_cpp_tests PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )

  include(GoogleTest)
  gtest_discover_tests(gaffa_cpp_tests)

  if(GAFFA_ENABLE_LOKI)
    add_executable(gaffa_loki_smoke_tests
      tests/loki/test_loki_smoke.cpp
    )
    target_link_libraries(gaffa_loki_smoke_tests PRIVATE
      GTest::gtest_main
      gaffa_loki
    )
    target_include_directories(gaffa_loki_smoke_tests PRIVATE
      include
      include/gaffa
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    set_target_properties(gaffa_loki_smoke_tests PROPERTIES
      BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      LINKER_LANGUAGE CUDA
    )
    gtest_discover_tests(gaffa_loki_smoke_tests)

    # gaffa_loki carries its core dependency transitively.
    add_executable(gaffa_loki_pffa_tests tests/loki/test_loki_pffa.cpp)
    target_link_libraries(gaffa_loki_pffa_tests PRIVATE
      GTest::gtest_main
      gaffa_loki
    )
    target_include_directories(gaffa_loki_pffa_tests PRIVATE
      include
      include/gaffa
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    set_target_properties(gaffa_loki_pffa_tests PROPERTIES
      BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      LINKER_LANGUAGE CUDA
    )
    gtest_discover_tests(gaffa_loki_pffa_tests)
  endif()

  add_executable(gaffa_search_config_tests
    cli/gaffa_search/config.cpp
    cli/gaffa_search/config_yaml.cpp
    tests/cli/test_gaffa_search_config.cpp
  )
  target_link_libraries(gaffa_search_config_tests PRIVATE
    GTest::gtest_main
    yaml-cpp::yaml-cpp
  )
  target_include_directories(gaffa_search_config_tests PRIVATE
    include
    cli/gaffa_search
  )
  gtest_discover_tests(gaffa_search_config_tests)

  add_executable(gaffa_search_output_tests
    cli/gaffa_search/config.cpp
    cli/gaffa_search/config_yaml.cpp
    cli/gaffa_search/report.cpp
    cli/gaffa_search/output.cpp
    tests/cli/test_gaffa_search_output.cpp
  )
  target_link_libraries(gaffa_search_output_tests PRIVATE
    GTest::gtest_main
    gaffa_core
    yaml-cpp::yaml-cpp
  )
  target_include_directories(gaffa_search_output_tests PRIVATE
    include
    cli/gaffa_search
  )
  set_target_properties(gaffa_search_output_tests PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )
  gtest_discover_tests(gaffa_search_output_tests)
endif()

file(GLOB GAFFA_BENCHMARK_SOURCES CONFIGURE_DEPENDS
  benchmarks/*_benchmark.cpp
)
# Loki benchmarks link an optional backend and therefore cannot participate in
# the always-available core benchmark loop.
list(FILTER GAFFA_BENCHMARK_SOURCES EXCLUDE REGEX ".*/loki_.*_benchmark\\.cpp$")
set(GAFFA_BENCHMARK_TARGETS)
foreach(benchmark_source IN LISTS GAFFA_BENCHMARK_SOURCES)
  get_filename_component(benchmark_name "${benchmark_source}" NAME_WE)
  set(benchmark_target "gaffa_${benchmark_name}")
  add_executable(${benchmark_target} EXCLUDE_FROM_ALL
    ${benchmark_source}
    src/gaffa/io/filterbank_legacy.cpp
  )
  target_link_libraries(${benchmark_target} PRIVATE gaffa_core)
  target_include_directories(${benchmark_target} PRIVATE include include/gaffa)
  target_compile_definitions(${benchmark_target}
    PRIVATE GAFFA_ENABLE_LEGACY_FILTERBANK_REFERENCE
  )
  set_target_properties(${benchmark_target} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )
  list(APPEND GAFFA_BENCHMARK_TARGETS ${benchmark_target})
endforeach()

if(GAFFA_ENABLE_LOKI)
  add_executable(gaffa_loki_dm_search_benchmark EXCLUDE_FROM_ALL
    benchmarks/loki_dm_search_benchmark.cpp
    src/gaffa/io/filterbank_legacy.cpp
  )
  target_link_libraries(gaffa_loki_dm_search_benchmark PRIVATE
    gaffa_loki
  )
  target_include_directories(gaffa_loki_dm_search_benchmark PRIVATE
    include
    include/gaffa
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_compile_definitions(gaffa_loki_dm_search_benchmark
    PRIVATE GAFFA_ENABLE_LEGACY_FILTERBANK_REFERENCE
  )
  set_target_properties(gaffa_loki_dm_search_benchmark PROPERTIES
    BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )
endif()

# The production command-line runner is an orchestration layer. Keep it
# outside src/gaffa so CLI configuration and device scheduling do not become
# part of the reusable algorithm library.
add_executable(gaffa_search
  cli/gaffa_search/config.cpp
  cli/gaffa_search/config_yaml.cpp
  cli/gaffa_search/cuda_worker.cpp
  cli/gaffa_search/output.cpp
  cli/gaffa_search/report.cpp
  cli/gaffa_search/search.cpp
  cli/gaffa_search/main.cpp
)
target_link_libraries(gaffa_search PRIVATE
  gaffa_core
  yaml-cpp::yaml-cpp
)
target_include_directories(gaffa_search PRIVATE include cli/gaffa_search)
set_target_properties(gaffa_search PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)
if(GAFFA_ENABLE_LOKI)
  target_link_libraries(gaffa_search PRIVATE gaffa_loki)
  target_compile_definitions(gaffa_search PRIVATE GAFFA_SEARCH_ENABLE_LOKI)
  set_target_properties(gaffa_search PROPERTIES
    BUILD_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
    INSTALL_RPATH "${GAFFA_LOKI_LIBRARY_DIR}"
  )
endif()

if(GAFFA_BENCHMARK_TARGETS)
  add_custom_target(gaffa_benchmarks DEPENDS ${GAFFA_BENCHMARK_TARGETS})
endif()

install(TARGETS _core LIBRARY DESTINATION gaffa)
if(GAFFA_ENABLE_LOKI)
  install(TARGETS _loki LIBRARY DESTINATION gaffa)
endif()

if(SKBUILD)
  set(GAFFA_RUNTIME_INSTALL_DIR "${SKBUILD_SCRIPTS_DIR}")
else()
  include(GNUInstallDirs)
  set(GAFFA_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
endif()
install(TARGETS gaffa_search RUNTIME DESTINATION "${GAFFA_RUNTIME_INSTALL_DIR}")

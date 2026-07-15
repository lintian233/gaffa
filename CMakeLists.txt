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
include(CTest)

# Loki is an optional periodic-search backend.  Keep its dependency graph out
# of the core library until the dedicated adapter target is introduced.
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

# CUDA memory is shared infrastructure. Keep it separate from the core object
# collection so optional backends can link it without pulling in gaffa_core.
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

if(GAFFA_ENABLE_LOKI)
  add_library(gaffa_loki STATIC
    src/gaffa/loki/loki_pffa_plan.cpp
    src/gaffa/loki/loki_pffa.cu
  )
  target_link_libraries(gaffa_loki PUBLIC
    gaffa_cuda_memory
    loki::loki
    CUDA::cudart
    CUDA::cuda_driver
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
  $<TARGET_OBJECTS:gaffa_core_objects>
)
target_link_libraries(_core PRIVATE gaffa_cuda_memory CUDA::cudart CUDA::cuda_driver)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(_core PRIVATE include)
set_target_properties(_core PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  file(GLOB_RECURSE GAFFA_CPP_TEST_SOURCES CONFIGURE_DEPENDS
    tests/cpp/test_*.cpp
  )

  add_executable(gaffa_cpp_tests
    src/gaffa/io/filterbank_legacy.cpp
    ${GAFFA_CPP_TEST_SOURCES}
    $<TARGET_OBJECTS:gaffa_core_objects>
  )
  target_link_libraries(gaffa_cpp_tests PRIVATE GTest::gtest_main gaffa_cuda_memory CUDA::cudart CUDA::cuda_driver)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gaffa_cpp_tests PRIVATE OpenMP::OpenMP_CXX)
  endif()
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
      $<TARGET_OBJECTS:gaffa_core_objects>
    )
    target_link_libraries(gaffa_loki_smoke_tests PRIVATE
      GTest::gtest_main
      gaffa_loki
      gaffa_cuda_memory
      CUDA::cudart
      CUDA::cuda_driver
    )
    if(OpenMP_CXX_FOUND)
      target_link_libraries(gaffa_loki_smoke_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
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

    # This target intentionally does not include gaffa_core_objects. It proves
    # that gaffa_loki carries every implementation dependency of its public API.
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
    $<TARGET_OBJECTS:gaffa_core_objects>
  )
  target_link_libraries(${benchmark_target} PRIVATE gaffa_cuda_memory CUDA::cudart CUDA::cuda_driver)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${benchmark_target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
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
    $<TARGET_OBJECTS:gaffa_core_objects>
  )
  target_link_libraries(gaffa_loki_dm_search_benchmark PRIVATE
    gaffa_loki
    gaffa_cuda_memory
    CUDA::cudart
    CUDA::cuda_driver
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gaffa_loki_dm_search_benchmark PRIVATE OpenMP::OpenMP_CXX)
  endif()
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

if(GAFFA_BENCHMARK_TARGETS)
  add_custom_target(gaffa_benchmarks DEPENDS ${GAFFA_BENCHMARK_TARGETS})
endif()

install(TARGETS _core LIBRARY DESTINATION gaffa)

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

file(GLOB_RECURSE GAFFA_CORE_SOURCES CONFIGURE_DEPENDS
  src/gaffa/*.cpp
  src/gaffa/*.cu
)
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/bindings\\.cpp$")
list(FILTER GAFFA_CORE_SOURCES EXCLUDE REGEX ".*/.*_legacy\\.cpp$")

add_library(gaffa_core_objects OBJECT ${GAFFA_CORE_SOURCES})
target_link_libraries(gaffa_core_objects PUBLIC CUDA::cudart)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gaffa_core_objects PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(gaffa_core_objects PUBLIC include)
set_target_properties(gaffa_core_objects PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

pybind11_add_module(_core
  src/gaffa/bindings.cpp
  $<TARGET_OBJECTS:gaffa_core_objects>
)
target_link_libraries(_core PRIVATE CUDA::cudart)
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
  target_link_libraries(gaffa_cpp_tests PRIVATE GTest::gtest_main CUDA::cudart)
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
endif()

file(GLOB GAFFA_BENCHMARK_SOURCES CONFIGURE_DEPENDS
  benchmarks/*_benchmark.cpp
)
set(GAFFA_BENCHMARK_TARGETS)
foreach(benchmark_source IN LISTS GAFFA_BENCHMARK_SOURCES)
  get_filename_component(benchmark_name "${benchmark_source}" NAME_WE)
  set(benchmark_target "gaffa_${benchmark_name}")
  add_executable(${benchmark_target}
    ${benchmark_source}
    src/gaffa/io/filterbank_legacy.cpp
    $<TARGET_OBJECTS:gaffa_core_objects>
  )
  target_link_libraries(${benchmark_target} PRIVATE CUDA::cudart)
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

if(GAFFA_BENCHMARK_TARGETS)
  add_custom_target(gaffa_benchmarks DEPENDS ${GAFFA_BENCHMARK_TARGETS})
endif()

install(TARGETS _core LIBRARY DESTINATION gaffa)
install(FILES python/gaffa/__init__.py DESTINATION gaffa)

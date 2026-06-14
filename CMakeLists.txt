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
include(CTest)

add_library(gaffa_core_objects OBJECT
  src/gaffa/vector_add.cpp
  src/gaffa/vector_add.cu
)
target_link_libraries(gaffa_core_objects PUBLIC CUDA::cudart)
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
target_include_directories(_core PRIVATE include)
set_target_properties(_core PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  add_executable(gaffa_cpp_tests
    tests/cpp/test_vector_add.cpp
    $<TARGET_OBJECTS:gaffa_core_objects>
  )
  target_link_libraries(gaffa_cpp_tests PRIVATE GTest::gtest_main CUDA::cudart)
  target_include_directories(gaffa_cpp_tests PRIVATE include)
  set_target_properties(gaffa_cpp_tests PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
  )

  include(GoogleTest)
  gtest_discover_tests(gaffa_cpp_tests)
endif()

install(TARGETS _core LIBRARY DESTINATION gaffa)
install(FILES python/gaffa/__init__.py DESTINATION gaffa)

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

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
  src/gaffa/bindings.cpp
  src/gaffa/vector_add.cu
)
target_link_libraries(_core PRIVATE CUDA::cudart)
target_include_directories(_core PRIVATE include)
set_target_properties(_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINKER_LANGUAGE CUDA
)

install(TARGETS _core LIBRARY DESTINATION gaffa)
install(FILES python/gaffa/__init__.py DESTINATION gaffa)

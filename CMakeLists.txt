cmake_minimum_required(VERSION 3.12)

project(CUDATestBed VERSION 0.1
                    DESCRIPTION "Personal CUDA test bed"
                    LANGUAGES CXX CUDA)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line
option(BUILD_TEST "whether build test for this project or not" ON)
option(ENABLE_OPENMP "whether use openmp" ON)
option(DEBUG "debug mode" OFF)


# https://stackoverflow.com/questions/23323741/building-of-executable-and-shared-library-with-cmake-runtimelinker-does-not-fin
# https://stackoverflow.com/questions/47251874/cmake-shared-library-in-subdirectory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# By default, new Visual Studio projects link the C runtimes dynamically but GoogleTest links them statically.
# We force google test to link C runtimes dynamically here.
# see https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

if (MSVC)
  # https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # multi-thread compile
endif (MSVC)

if (ENABLE_OPENMP)
  add_definitions(-DUSE_OPENMP)
  message(STATUS "Enable OpenMP")
  if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  else (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif (MSVC)
endif (ENABLE_OPENMP)

# use cuda
include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
message(STATUS "Use CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID}:${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Compiler Path: ${CMAKE_CUDA_COMPILER}")

# find cuda path
if(NOT CUDA_TOOLKIT_ROOT_DIR)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
endif()
message(STATUS "CUDA ROOT Path: ${CUDA_TOOLKIT_ROOT_DIR}")

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda") # enable lambda in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # enable constexpr expression in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUDA_ARCHITECTURES}")
message(STATUS "Build with CUDA Computation Capability: ${CUDA_ARCHITECTURES}")

# cuRAND
include("${CMAKE_SOURCE_DIR}/cmake/modules/cuRAND.cmake")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# tensor library
# TOUCH: https://stackoverflow.com/questions/1027247/is-it-better-to-specify-source-files-with-glob-or-each-file-individually-in-cmak/1060061#1060061
file(TOUCH_NOCREATE "${CMAKE_SOURCE_DIR}/src/tensor/CMakeLists.txt")
add_subdirectory("${CMAKE_SOURCE_DIR}/src/tensor")

# matmul
file(TOUCH_NOCREATE "${CMAKE_SOURCE_DIR}/src/matmul/CMakeLists.txt")
add_subdirectory("${CMAKE_SOURCE_DIR}/src/matmul")

if(BUILD_TEST)
  message(STATUS "Build with unittest")
  add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/googletest")
  enable_testing()
  include(GoogleTest)
  message(STATUS "Find google test in ${gtest_SOURCE_DIR}")
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories("${PROJECT_SOURCE_DIR}/include")
  include_directories("${CMAKE_SOURCE_DIR}/third_party/FP16/include")
  file(GLOB_RECURSE TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/test/cpp/*.cc")
  add_executable(runUnitTests ${TEST_SRC_FILES})
  add_dependencies(runUnitTests tensor)
  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests tensor)
  add_test(UnitTests runUnitTests)
endif()

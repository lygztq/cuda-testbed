cmake_minimum_required(VERSION 3.12)

project(CUDATestBed VERSION 0.1
                    DESCRIPTION "Personal CUDA test bed"
                    LANGUAGES CXX CUDA)

option(BUILD_TEST "whether build test for this project or not" ON)
option(DEBUG "debug mode" OFF)

# By default, new Visual Studio projects link the C runtimes dynamically but GoogleTest links them statically.
# We force google test to link C runtimes dynamically here.
# see https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# use cuda
include(CheckLanguage)
check_language(CUDA)
message(STATUS "Use CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID}:${CMAKE_CUDA_COMPILER_VERSION}")

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# tensor library
# TOUCH: https://stackoverflow.com/questions/1027247/is-it-better-to-specify-source-files-with-glob-or-each-file-individually-in-cmak/1060061#1060061
file(TOUCH_NOCREATE "${CMAKE_SOURCE_DIR}/src/tensor/CMakeLists.txt")
add_subdirectory("${CMAKE_SOURCE_DIR}/src/tensor")

if(BUILD_TEST)
  message(STATUS "Build with unittest")
  add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/googletest")
  enable_testing()
  include(GoogleTest)
  message(STATUS "Find google test in ${gtest_SOURCE_DIR}")
  include_directories("${gtest_SOURCE_DIR}/include")
  include_directories("${PROJECT_SOURCE_DIR}/include")
  file(GLOB_RECURSE TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/test/cpp/tensor/*.cc")
  add_executable(runUnitTests ${TEST_SRC_FILES})
  add_dependencies(runUnitTests tensor)
  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests tensor)
  add_test(UnitTests runUnitTests)
endif()

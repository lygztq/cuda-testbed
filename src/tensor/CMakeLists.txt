add_definitions(-DTENSOR_EXPORTS)

file(GLOB_RECURSE TENSOR_SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/tensor/*.cc")
add_library(tensor SHARED ${TENSOR_SOURCE_LIST})

target_include_directories(tensor PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_features(tensor PUBLIC cxx_std_17)
target_link_libraries(tensor CUDART)

add_definitions(-DTENSOR_EXPORTS)

file(GLOB_RECURSE TENSOR_SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/tensor/*.cc")
file(GLOB_RECURSE TENSOR_CU_SOURCE_LIST "${CMAKE_SOURCE_DIR}/src/tensor/*.cu")
list(APPEND TENSOR_SOURCE_LIST ${TENSOR_CU_SOURCE_LIST})
add_library(tensor SHARED ${TENSOR_SOURCE_LIST})

set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tensor PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
target_include_directories(tensor PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(tensor PUBLIC "${CMAKE_SOURCE_DIR}/third_party/FP16/include")
target_compile_features(tensor PUBLIC cxx_std_17)
target_link_libraries(tensor CUDART)

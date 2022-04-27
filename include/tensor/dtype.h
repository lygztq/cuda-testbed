#ifndef TENSOR_DTYPE_H_
#define TENSOR_DTYPE_H_

#include "tensor/fp16.h"

namespace tensor {

enum class DataType : size_t {
  kInt8,
  kUInt8,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kHalf,
  kFloat,
  kDouble
};



} // namespace tensor

#endif  // TENSOR_DTYPE_H_

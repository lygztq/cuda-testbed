#ifndef TENSOR_DTYPE_H_
#define TENSOR_DTYPE_H_

#include <cstdint>
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
  kDouble,
};

template <typename CRT>
struct CRuntimeTypeToDataType {};

#define DEFINE_CRT_TO_DTYPE(CRT, DType)   \
template <>                               \
struct CRuntimeTypeToDataType<CRT> {      \
  static constexpr DataType type = DType; \
};

DEFINE_CRT_TO_DTYPE(int8_t, DataType::kInt8);
DEFINE_CRT_TO_DTYPE(uint8_t, DataType::kUInt8);
DEFINE_CRT_TO_DTYPE(int32_t, DataType::kInt32);
DEFINE_CRT_TO_DTYPE(uint32_t, DataType::kUInt32);
DEFINE_CRT_TO_DTYPE(int64_t, DataType::kInt64);
DEFINE_CRT_TO_DTYPE(uint64_t, DataType::kUInt64);
DEFINE_CRT_TO_DTYPE(fp16_t, DataType::kHalf);
DEFINE_CRT_TO_DTYPE(float, DataType::kFloat);
DEFINE_CRT_TO_DTYPE(double, DataType::kDouble);

#undef DEFINE_CRT_TO_DTYPE

template <typename CRT>
struct SupportCRT {
  static constexpr bool value = false;
};

#define DEFINE_SUPPORT_CRT(CRT)         \
template <>                             \
struct SupportCRT<CRT> {                \
  static constexpr bool value = true;   \
};

DEFINE_SUPPORT_CRT(int8_t)
DEFINE_SUPPORT_CRT(int32_t)
DEFINE_SUPPORT_CRT(int64_t)
DEFINE_SUPPORT_CRT(uint8_t)
DEFINE_SUPPORT_CRT(uint32_t)
DEFINE_SUPPORT_CRT(uint64_t)
DEFINE_SUPPORT_CRT(fp16_t)
DEFINE_SUPPORT_CRT(float)
DEFINE_SUPPORT_CRT(double)

#undef DEFINE_SUPPORT_CRT

template <typename CRT>
using crt_to_dtype_t = typename CRuntimeTypeToDataType<CRT>::type;

inline size_t DataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8: case DataType::kUInt8:
      return 1;
    case DataType::kHalf:
      return 2;
    case DataType::kInt32: case DataType::kUInt32: case DataType::kFloat:
      return 4;
    case DataType::kInt64: case DataType::kUInt64: case DataType::kDouble:
      return 8;
    default:
      return 0;
  }
}

} // namespace tensor

#endif  // TENSOR_DTYPE_H_

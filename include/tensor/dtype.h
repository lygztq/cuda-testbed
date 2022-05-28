/*!
 * \file dtype.h
 * \brief Data types supported by Tensor.
 */
#ifndef TENSOR_DTYPE_H_
#define TENSOR_DTYPE_H_

#include <cstdint>
#include <type_traits>
#include <variant>
#include "tensor/fp16.h"
#include "tensor/device.h"

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
constexpr inline DataType crt_to_dtype_v = CRuntimeTypeToDataType<CRT>::type;

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
DEFINE_SUPPORT_CRT(uint16_t) // for pure byte computation
DEFINE_SUPPORT_CRT(float)
DEFINE_SUPPORT_CRT(double)

#undef DEFINE_SUPPORT_CRT
template <typename T>
inline constexpr bool support_crt_v = SupportCRT<T>::value;

template <typename T1, typename T2, DeviceType XPU>
struct dtype_cast {};

template <typename T1, typename T2>
struct dtype_cast<T1, T2, DeviceType::kCPU> {
  static T2 cast(T1 src) { return static_cast<T2>(src); }
};

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

#define DTYPE_SWITCH_CASE(switch_t, crt, ...) \
  case switch_t: {                            \
    using scalar_t = crt;                     \
    __VA_ARGS__();                            \
    break;                                    \
  }

#define DTYPE_SWITCH(dtype, ...)                                \
  DataType _st = DataType(dtype);                               \
  switch (_st) {                                                \
    DTYPE_SWITCH_CASE(DataType::kInt8,   int8_t, __VA_ARGS__)   \
    DTYPE_SWITCH_CASE(DataType::kUInt8,  uint8_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kInt32,  int32_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kUInt32, uint32_t, __VA_ARGS__) \
    DTYPE_SWITCH_CASE(DataType::kInt64,  int64_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kUInt64, uint64_t, __VA_ARGS__) \
    DTYPE_SWITCH_CASE(DataType::kHalf,   fp16_t, __VA_ARGS__)   \
    DTYPE_SWITCH_CASE(DataType::kFloat,  float, __VA_ARGS__)    \
    DTYPE_SWITCH_CASE(DataType::kDouble, double, __VA_ARGS__)   \
  }

#define DTYPE_SWITCH_WITHOUT_HALF(dtype, ...)                   \
  DataType _st = DataType(dtype);                               \
  switch (_st) {                                                \
    DTYPE_SWITCH_CASE(DataType::kInt8,   int8_t, __VA_ARGS__)   \
    DTYPE_SWITCH_CASE(DataType::kUInt8,  uint8_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kInt32,  int32_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kUInt32, uint32_t, __VA_ARGS__) \
    DTYPE_SWITCH_CASE(DataType::kInt64,  int64_t, __VA_ARGS__)  \
    DTYPE_SWITCH_CASE(DataType::kUInt64, uint64_t, __VA_ARGS__) \
    DTYPE_SWITCH_CASE(DataType::kFloat,  float, __VA_ARGS__)    \
    DTYPE_SWITCH_CASE(DataType::kDouble, double, __VA_ARGS__)   \
  }

#define DTYPE_SWITCH_FLOAT(dtype, ...)                   \
  DataType _st = DataType(dtype);                               \
  switch (_st) {                                                \
    DTYPE_SWITCH_CASE(DataType::kHalf,   fp16_t, __VA_ARGS__)   \
    DTYPE_SWITCH_CASE(DataType::kFloat,  float, __VA_ARGS__)    \
    DTYPE_SWITCH_CASE(DataType::kDouble, double, __VA_ARGS__)   \
  }

template <typename T>
struct is_floatint_point {
  static constexpr bool value = false;
};

#define IS_FLOATINT_POINT_CASE(type)  \
template <>                           \
struct is_floatint_point<type> {      \
  static constexpr bool value = true; \
}
IS_FLOATINT_POINT_CASE(float);
IS_FLOATINT_POINT_CASE(double);
IS_FLOATINT_POINT_CASE(fp16_t);
#undef IS_FLOATINT_POINT_CASE

template <typename T>
struct is_signed_integral {
  static constexpr bool value = false;
};

#define IS_SIGNED_INTEGRAL_CASE(type) \
template <>                           \
struct is_signed_integral<type> {     \
  static constexpr bool value = true; \
}
IS_SIGNED_INTEGRAL_CASE(int8_t);
IS_SIGNED_INTEGRAL_CASE(int32_t);
IS_SIGNED_INTEGRAL_CASE(int64_t);
#undef IS_SIGNED_INTEGRAL_CASE

template <typename T>
struct is_unsigned_integral {
  static constexpr bool value = false;
};

#define IS_UNSIGNED_INTEGRAL_CASE(type) \
template <>                             \
struct is_unsigned_integral<type> {     \
  static constexpr bool value = true;   \
}
IS_UNSIGNED_INTEGRAL_CASE(uint8_t);
IS_UNSIGNED_INTEGRAL_CASE(uint32_t);
IS_UNSIGNED_INTEGRAL_CASE(uint64_t);
#undef IS_UNSIGNED_INTEGRAL_CASE

class Scalar {
public:
  template <typename T>
  Scalar(T val) : val_(val) {}

  template <typename T>
  T To() const {
    return std::visit([](auto&& arg) -> T {
      return static_cast<T>(arg);
    }, val_);
  }

#define DEFINE_CAST(type) \
  operator type() const { \
    return To<type>(); \
  }

DEFINE_CAST(float)
DEFINE_CAST(double)
DEFINE_CAST(int8_t)
DEFINE_CAST(uint8_t)
DEFINE_CAST(int32_t)
DEFINE_CAST(uint32_t)
DEFINE_CAST(int64_t)
DEFINE_CAST(uint64_t)

private:
    std::variant<
    int8_t, uint8_t,
    int32_t, uint32_t,
    int64_t, uint64_t,
    fp16_t, float, double> val_;
};

} // namespace tensor

#endif  // TENSOR_DTYPE_H_

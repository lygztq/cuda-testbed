/*!
 * \file common_funcs.h
 * \brief commonly used inline functions
 */
#ifndef TENSOR_COMMON_FUNCS_H_
#define TENSOR_COMMON_FUNCS_H_

#include <vector>
#include <type_traits>
#include <numeric>

namespace tensor {
namespace common {

template <typename T, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
inline size_t ShapeNumElem(const std::vector<T>& shape) {
  if (shape.empty()) return 0;
#ifdef __GNUC__
    return static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>{}));
#else // __GNUC__
    return static_cast<size_t>(std::reduce(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>{}));
#endif // __GNUC__
}

template <typename T, size_t N, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
inline size_t ShapeNumElem(const std::array<T, N>& arr_shape, const size_t num_axes) {
  if (num_axes == 0) return 0;
#ifdef __GNUC__
    return static_cast<size_t>(std::accumulate(arr_shape.cbegin(), arr_shape.cbegin() + num_axes, static_cast<T>(1), std::multiplies<T>{}));
#else // __GNUC__
    return static_cast<size_t>(std::reduce(arr_shape.cbegin(), arr_shape.cbegin() + num_axes, static_cast<T>(1), std::multiplies<T>{}));
#endif // __GNUC__
}

} // namespace common
} // namespace tensor


#endif  // TENSOR_COMMON_FUNCS_H_

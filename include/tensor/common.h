/*!
 * \file common.h
 * \brief commonly used things
 */
#ifndef TENSOR_COMMON_H_
#define TENSOR_COMMON_H_

#include <vector>
#include <type_traits>
#include <numeric>
#include "tensor/macros.h"

namespace tensor {

/* -- constants -- */

// Max number of axes (a.k.a dimensions) a tensor can have
constexpr size_t kMaxTensorAxis = 16;

namespace common {

/*!
 * \brief Deduce actual shape from a given number of elements and shape.
 *        e.g. num_elem = 12, shape = {-1, 2, 3} => result = {2, 2, 3}
 *
 * \param num_elem The total number of elements
 * \param shape The given shape to be deduced
 * \return The deduced shape
 */
TENSOR_DLL std::vector<size_t> ShapeDeduction(size_t num_elem, const std::vector<int>& shape);

/*!
 * \brief Computate total number of elements from a given shape (vector).
 *
 * \param shape The given shape
 * \return size_t Total number of elements
 */
template <typename T, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
inline size_t ShapeNumElem(const std::vector<T>& shape) {
  if (shape.empty())
    return 0;
#ifdef __GNUC__
  return static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>{}));
#else  // __GNUC__
  return static_cast<size_t>(std::reduce(shape.cbegin(), shape.cend(), static_cast<T>(1), std::multiplies<T>{}));
#endif // __GNUC__
}

/*!
 * \brief Computate total number of elements from a given shape (array).
 *
 * \param arr_shape The shape array with size equals to 'kMaxTensorAxis'
 * \param num_axes The actual number of axes of this shape
 * \return size_t Total number of elements
 */
template <typename T, size_t N, typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
inline size_t ShapeNumElem(const std::array<T, N>& arr_shape, const size_t num_axes) {
  if (num_axes == 0)
    return 0;
#ifdef __GNUC__
  return static_cast<size_t>(std::accumulate(arr_shape.cbegin(), arr_shape.cbegin() + num_axes, static_cast<T>(1), std::multiplies<T>{}));
#else  // __GNUC__
  return static_cast<size_t>(std::reduce(arr_shape.cbegin(), arr_shape.cbegin() + num_axes, static_cast<T>(1), std::multiplies<T>{}));
#endif // __GNUC__
}

} // namespace common
} // namespace tensor


#endif  // TENSOR_COMMON_H_

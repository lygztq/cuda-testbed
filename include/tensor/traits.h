#ifndef TENSOR_TRAITS_H_
#define TENSOR_TRAITS_H_

#include <type_traits>
#include <tuple>
#include <utility>
#include "tensor/macros.h"

namespace tensor {

// strip operator() function from a class type
template <typename T>
struct strip_class {};

template <typename C, typename R, typename... Args>
struct strip_class<R (C::*)(Args...)> {
  using type = R(Args...);
};

template <typename C, typename R, typename... Args>
struct strip_class<R (C::*)(Args...) const> {
  using type = R(Args...);
};

template <typename T>
using strip_class_t = typename strip_class<T>::type;

// Fallback, functor
template <typename T>
struct function_traits : public function_traits<strip_class_t<decltype(&T::operator())>> {};

// trivial function
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  static constexpr size_t arity = sizeof...(Args);
  using return_t = R;
  using args_t_tuple = typename std::tuple<Args...>;
  
  template <size_t i>
  struct arg {
    using type = typename std::tuple_element_t<i, args_t_tuple>;
  };

  template <size_t i>
  using arg_t = typename arg<i>::type;
};

// function pointer
template <typename R, typename... Args>
struct function_traits<R*(Args...)> : public function_traits<R(Args...)> {};

// remove reference and pointer
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

template <typename T>
struct is_function_traits {
  static constexpr bool value = false;
};

template <typename T>
struct is_function_traits<function_traits<T>> {
  static constexpr bool value = true;
};

// deference trait for argument tuple in function f(A1, A2, ...)
template <typename traits, size_t... INDEX>
typename traits::args_t_tuple
deference_impl(
  char* data[],
  const size_t* strides,
  size_t i,
  std::index_sequence<INDEX...> index) {
  return std::make_tuple(*(reinterpret_cast<typename traits::arg_t<INDEX>*>(
    data[INDEX] + i * strides[INDEX]))...);
}

template <typename traits, std::enable_if_t<is_function_traits<traits>::value, bool> = false>
typename traits::args_t_tuple
deference(char* data[], const size_t* strides, size_t i) {
  using index_t = std::make_index_sequence<traits::arity>;
  return deference_impl<traits>(data, strides, i, index_t{});
}

} // namespace tensor

#endif  // TENSOR_TRAITS_H_

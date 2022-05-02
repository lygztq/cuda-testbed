//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

// Modification from Pytorch:
// - modified from llvm::function_ref
// - added more SFINAE to enable use in overloaded functions

#ifndef TENSOR_FUNCTION_REF_H_
#define TENSOR_FUNCTION_REF_H_

#include <type_traits>
#include "tensor/traits.h"

namespace tensor {

/*!
 * \brief An efficient, type-erasing, non-owning reference to a callable. This is
 *        intended for use as the type of a function parameter that is not used
 *        after the function in question returns.
 * \note This class does not own the callable, so it is not in general safe to store
 *       a function_ref.
 */
template <typename T>
struct function_ref {};

template <typename R, typename... Args>
struct function_ref<R(Args...)> {
  using CallbackFnType = R(intptr_t, Args...);
  using Self = function_ref<R(Args...)>;
  CallbackFnType* callback_ = nullptr;
  intptr_t fn_ptr_;

  template <typename Callable>
  static R CallbackFn(intptr_t fn_ptr, Args... args) {
    return (*reinterpret_cast<Callable*>(fn_ptr))(std::forward<Args>(args)...);
  }

  function_ref() = default;
  function_ref(std::nullptr_t) {}

  template <typename Callable,
    typename std::enable_if_t<!std::is_same_v<
      std::remove_reference_t<Callable>, function_ref>>* = nullptr,
    typename std::enable_if_t<std::is_convertible_v<
      std::invoke_result_t<Callable&&, Args&&...>, R>>* = nullptr>
  function_ref(Callable&& callable)
    : fn_ptr_(reinterpret_cast<intptr_t>(&callable))
    , callback_(CallbackFn<std::remove_reference_t<Callable>>) {}

  R operator()(Args... args) {
    return callback_(fn_ptr_, std::forward<Args>(args)...);
  }

  operator bool() const {
    return callback_;
  }
};

// TODO: this is invalid (the static function CallbackFn will touch an invalid addredd, why?)
// template <typename Callable>
// decltype(auto) MakeFunctionRef(Callable* callable) {
//   return function_ref<Callable>(callable);
// }

} // namespace tensor

#endif  // TENSOR_FUNCTION_REF_H_

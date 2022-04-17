#ifndef TENSOR_LOOP_H_
#define TENSOR_LOOP_H_

#include <array>
#include <algorithm>
#include "tensor/traits.h"

namespace tensor {
namespace cpu {

template <typename Op>
struct Loop2d {
  Op op_;
  using op_trait = function_traits<Op>;
  static constexpr size_t ntensors = op_trait::arity + 1;
  using data_t = std::array<char*, ntensors>;

  explicit Loop2d(const Op& op) : op_(op) {}
  explicit Loop2d(Op&& op) : op_(std::move(op)) {}

  static void advance(data_t& data, const size_t* outer_strides) {
    for (auto i = 0; i < data.size(); ++i) {
      data[i] += outer_strides[i];
    }
  }

  void operator()(char** dptrs,
                  const size_t* strides,
                  size_t inner_size,
                  size_t outer_size) {
    data_t data;
    std::copy_n(dptrs, ntensors, data.data());
    const size_t* inner_strides = &strides[ntensors];
    const size_t* outer_strides = &strides[0];

    for (size_t outer = 0; outer < outer_size; ++outer) {
      auto optr = reinterpret_cast<op_trait::return_t*>(data[0]);
      for (size_t inner = 0; inner < inner_size; ++inner) {
        optr[inner] = std::apply(op_, deference<op_trait>(&data[1], &inner_strides[1], inner));
      }
      advance(data, outer_strides);
    }
  }
};
  
} // namespace cpu
} // namespace tensor


#endif  // TENSOR_LOOP_H_

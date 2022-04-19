#ifndef TENSOR_LOOP_H_
#define TENSOR_LOOP_H_

#include <array>
#include <algorithm>
#include "tensor/traits.h"

namespace tensor {
namespace cpu {

template <typename Op>
void BasicLoopFunc(const Op &op,
                   char** dptrs,
                   const size_t* strides,
                   size_t N) {
  using trait = function_traits<Op>;
  constexpr size_t ntensors = trait::arity + 1;

  // we assume the first tensor is the output tensor
  auto optr = reinterpret_cast<trait::return_t*>(dptrs[0]);

#ifdef _MSC_VER 
#pragma loop(hint_parallel(4))
#else // _MSC_VER
#pragma unroll
#endif //_MSC_VER
  for (size_t i = 0; i < N; ++i) {
    optr[i] = std::apply(op, deference<trait>(&dptrs[1], &strides[1], i));
  }
}

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
      BasicLoopFunc<Op>(op_, data.data(), inner_strides, inner_size);
      advance(data, outer_strides);
    }
  }
};
  
} // namespace cpu
} // namespace tensor


#endif  // TENSOR_LOOP_H_

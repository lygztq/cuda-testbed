#ifndef TENSOR_CPU_OP_H_
#define TENSOR_CPU_OP_H_

#include "tensor/iterator.h"
#include "tensor/loop.h"
#include "tensor/dtype.h"
#include "tensor/macros.h"

namespace tensor {
namespace ops {
namespace cpu {

template <typename Op>
void CPUContiguousKernel(TensorIterator& iter, Op&& elem_op) {
  CHECK(iter.Valid());
  using traits = function_traits<Op>;
  size_t num_elem = iter.NumElem();
  constexpr size_t num_tensors = traits::arity + 1;

  std::array<char*, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;

#ifndef _MSC_VER
  #pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char*>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].ElemSize();
  }

  BasicLoopFunc<Op>(elem_op, base_ptrs.data(), elem_sizes.data(), num_elem);
}

template <typename Op>
void CPUElemwiseKernel(TensorIterator& iter, Op&& elem_op) {
  Loop2d<Op> loop = MakeLoop2d(std::forward<Op>(elem_op));
  iter.ForEach(loop);
}

// This copy function will follow the data layout
// of the destination tensor.
TENSOR_DLL void ElemwiseCopyKernel(const Tensor& src, Tensor& dst);

// This copy function will ignore and overwrite
// the data layout of the destination tensor.
TENSOR_DLL void CopyKernel(const Tensor& src, Tensor& dst);

TENSOR_DLL void CastCopyKernel(const Tensor& src, Tensor& dst);

template <typename T, typename std::enable_if_t<support_crt_v<T>>* = nullptr>
void FillKernel(Tensor& tensor, T val) {
  T* dptr = tensor.TypedPtr<T>();
  size_t num_elem = tensor.NumElem();

// TODO: use openmp here
#ifdef _MSC_VER
#pragma loop(hint_parallel(4))
#endif //_MSC_VER
  for (size_t i = 0; i < num_elem; ++i) {
    dptr[i] = val;
  }
}

// TENSOR_DLL Tensor ContiguousKernel(const Tensor& src);

} // namespace cpu  
} // namespace ops
} // namespace tensor


#endif  // TENSOR_CPU_OP_H_

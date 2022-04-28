#ifndef TENSOR_CPU_OP_IMPL_H_
#define TENSOR_CPU_OP_IMPL_H_

#include "tensor/iterator.h"
#include "tensor/loop.h"
#include "tensor/dtype.h"
#include "tensor/macros.h"

namespace tensor {
namespace ops {
namespace cpu {

template <typename Op>
void CPUKernel(TensorIterator& iter, Op&& elem_op) {
  Loop2d loop = MakeLoop2d(std::forward<Op>(elem_op));
  iter.ForEach(loop);
}

// This copy function will follow the data layout
// of the destination tensor.
TENSOR_DLL void ElemwiseCopyKernel(const Tensor& src, Tensor& dst);

// This copy function will ignore and overwrite
// the data layout of the destination tensor.
TENSOR_DLL void CopyKernel(const Tensor& src, Tensor& dst);

TENSOR_DLL Tensor ContiguousKernel(const Tensor& src);

} // namespace cpu  
} // namespace ops
} // namespace tensor


#endif  // TENSOR_CPU_OP_IMPL_H_

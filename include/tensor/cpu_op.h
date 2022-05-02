/*!
 * \file cpu_op.h
 * \brief CPU implementation of tensor operator kernels
 */
#ifndef TENSOR_CPU_OP_H_
#define TENSOR_CPU_OP_H_

#include "tensor/iterator.h"
#include "tensor/loop.h"
#include "tensor/dtype.h"
#include "tensor/macros.h"

namespace tensor {
namespace ops {
namespace cpu {

/*!
 * \brief Element-wise operation on \b contiguous tensors.
 *
 * \tparam Op The type of \a 'elem_op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param elem_op A callable object. The element-wise operation to be performed.
 */
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

/*!
 * \brief Element-wise operation on tensors.
 *
 * \tparam Op The type of 'elem_op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param elem_op A callable object. The element-wise operation to be performed.
 */
template <typename Op>
void CPUElemwiseKernel(TensorIterator& iter, Op&& elem_op) {
  Loop2d<Op> loop = MakeLoop2d(std::forward<Op>(elem_op));
  iter.ForEach(loop);
}

/*!
 * \brief Copy elements from \a `src` tensor to \a 'dst' tensor.
 *        Src and dst must in the same device and with the same dtype.
 *
 * \param src The source tensor
 * \param dst The destination tensor
 * \note This copy function will follow the data
 *       layout of the destination tensor.
 */
TENSOR_DLL void ElemwiseCopyKernel(const Tensor& src, Tensor& dst);

/*!
 * \brief Copy elements from \a `src` tensor to \a 'dst' tensor.
 *        Src and dst must in the same device and with the same dtype.
 *
 * \param src The source tensor
 * \param dst The destination tensor
 * \note This copy function will ignore and overwrite
 *       the data layout of the destination tensor.
 */
TENSOR_DLL void CopyKernel(const Tensor& src, Tensor& dst);

/*!
 * \brief Copy elements in \a 'src' to \a 'dst'. Cast the dtype
 *        if needed.
 * 
 * \param src The source tensor
 * \param dst The destination tensor
 * \note This kernel is only valid on tensors with the same shape and stride.
 */
TENSOR_DLL void CastCopyKernel(const Tensor& src, Tensor& dst);

/*!
 * \brief Fill a tensor with the given value.
 * 
 * \tparam T Type of the given value.
 * \param tensor The tensor to be filled.
 * \param val The value used for filling tensors.
 * \note Type \a T must match dtype of the given tensor.
 */
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

} // namespace cpu
}   // namespace ops
} // namespace tensor

#endif // TENSOR_CPU_OP_H_

/*!
 * \file cuda_op.h
 * \brief CUDA implementation of tensor operator kernels
 */
#ifndef TENSOR_CUDA_OP_H_
#define TENSOR_CUDA_OP_H_

#include <type_traits>
#include <array>
#include "tensor/dtype.h"
#include "tensor/tensor.h"
#include "tensor/iterator.h"
#include "tensor/cuda_tools.cuh"
#include "tensor/cuda_common.h"

namespace tensor {
namespace ops {
namespace cuda {

/*!
 * \brief Element-wise operation on \b contiguous tensors.
 *
 * \tparam Op The type of \a 'op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 * \note This kernel is for contiguous tensors
 */
template <typename Op>
void CUDAContiguousKernel(TensorIterator& iter, Op&& op) {
  CHECK(iter.Valid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

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

  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;
  cudaElemwiseKernelImpl<128, unroll>(iter.NumElem(), [=]CUDA_LAMBDA(size_t idx){
    size_t offset[traits::arity + 1];
    for (int i = 0; i < traits::arity + 1; ++i) {
      offset[i] = idx * elem_sizes[i];
    }
    arg0_t* out = (arg0_t*)(base_ptrs[0] + offset[0]);
    *out = invoke(op, &base_ptrs[1], &offset[1], 1);
  });
}

/*!
 * \brief Element-wise operation on tensors.
 *
 * \tparam Op The type of \a 'op'
 * \param iter The tensor iterator used for element-wise loop.
 * \param op A callable object. The element-wise operation to be performed.
 */
template <typename Op>
void CUDAElemwiseKernel(TensorIterator& iter, Op&& op) {
  CHECK(iter.Valid());
  using traits = function_traits<Op>;
  using arg0_t = typename traits::return_t;

  constexpr size_t num_tensors = traits::arity + 1;
  std::array<char*, num_tensors> base_ptrs;
  std::array<size_t, num_tensors> elem_sizes;
  std::vector<size_t> strides(num_tensors * iter.NumAxes()); // (num_axes * num_tensors)
#ifndef _MSC_VER
  #pragma unroll
#endif
  for (size_t t = 0; t < num_tensors; ++t) {
    base_ptrs[t] = reinterpret_cast<char*>(iter.Tensors()[t].RawPtr());
    elem_sizes[t] = iter.Tensors()[t].ElemSize();
  }

  for (size_t i = 0; i < iter.NumAxes(); ++i) {
#ifndef _MSC_VER
    #pragma unroll
#endif
    for (size_t t = 0; t < num_tensors; ++t) {
      strides[i * num_tensors + t] = iter.Tensors()[t].Stride(i);
    }
  }

  OffsetCalculator<num_tensors> offset_calc(
    iter.NumAxes(), iter.Shape(), strides, elem_sizes);
  constexpr size_t unroll = sizeof(arg0_t) >= 4 ? 2 : 4;
  cudaElemwiseKernelImpl<128, unroll>(iter.NumElem(), [=]CUDA_LAMBDA(size_t idx){
    size_t offset[traits::arity + 1];
    offset_calc.get(idx, &offset[0]);
    arg0_t* out = (arg0_t*)(base_ptrs[0] + offset[0]);
    *out = invoke(op, &base_ptrs[1], &offset[1], 1);
  });
}

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
 * \brief Fill a tensor with the given value.
 *
 * \tparam T Type of the given value.
 * \param tensor The tensor to be filled.
 * \param val The value used for filling tensors.
 * \note Type \a T must match dtype of the given tensor.
 */
template <typename T, typename std::enable_if_t<support_crt_v<T>>* = nullptr>
TENSOR_DLL void FillKernel(Tensor& tensor, T val);

/*!
 * \brief Copy elements in \a 'src' to \a 'dst'. Cast the dtype
 *        if needed.
 *
 * \param src The source tensor
 * \param dst The destination tensor
 * \note This kernel is only valid on tensors with the same shape and stride.
 */
TENSOR_DLL void CastCopyKernel(const Tensor& src, Tensor& dst);

TENSOR_DLL void RandomUniformKernel(Tensor& tensor, Scalar low, Scalar high);

TENSOR_DLL void RandomNormalKernel(Tensor& tensor, Scalar mean, Scalar stddev);

} // namespace cuda
} // namespace ops
} // namespace tensor

#endif  // TENSOR_CUDA_OP_H_

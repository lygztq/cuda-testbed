#include "tensor/cuda_op.h"
#include "tensor/dtype.h"

namespace tensor {
namespace ops {
namespace cuda {

// This copy function will ignore and overwrite
// the data layout of the destination tensor.
void CopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  CHECK_EQ(src.GetDevice().type, DeviceType::kCUDA);

  TensorIterator iter;
  iter.AddInput(src);
  iter.AddOutput(dst);
  iter.Build();

  DTYPE_SWITCH(src.GetDataType(), [&](){
    CUDAElemwiseKernel(iter, [] CUDA_LAMBDA (scalar_t elem) { return elem; });
  });
}

template <typename T, typename std::enable_if_t<support_crt_v<T>>*>
void FillKernel(Tensor& tensor, T val) {
  T* dptr = tensor.TypedPtr<T>();
  constexpr size_t unroll = sizeof(T) >= 4 ? 2 : 4;
  FillKernelImpl<128, unroll>(tensor.NumElem(), dptr, val);
}

template void FillKernel<uint8_t, nullptr>(Tensor& t, uint8_t val);
template void FillKernel<uint16_t, nullptr>(Tensor& t, uint16_t val);
template void FillKernel<uint32_t, nullptr>(Tensor& t, uint32_t val);
template void FillKernel<uint64_t, nullptr>(Tensor& t, uint64_t val);

void CastCopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  CHECK_EQ(src.GetDevice().type, DeviceType::kCUDA);

  TensorIterator iter;
  iter.AddInput(src);
  iter.AddOutput(dst);
  iter.Build();

  DTYPE_SWITCH(dst.GetDataType(), [&](){
    using dst_t = scalar_t;
    DTYPE_SWITCH(src.GetDataType(), [&](){
      CUDAElemwiseKernel(iter, [] CUDA_LAMBDA (scalar_t elem) -> dst_t { return dtype_cast<scalar_t, dst_t, DeviceType::kCUDA>::cast(elem); });
    });
  });
}

} // namespace cuda
} // namespace ops
} // namespace tensor

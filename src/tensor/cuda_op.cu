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
  Device::SetCurrentDevice(src.GetDevice());

  TensorIterator iter;
  iter.AddInput(src);
  iter.AddOutput(dst);
  iter.Build();

  // TODO: change this to DTYPE_SWITCH_CUDA?
  DTYPE_SWITCH(src.GetDataType(), [&](){
    CUDAElemwiseKernel(iter, [] CUDA_LAMBDA (scalar_t elem) { return elem; });
  });
}

template <typename T, typename std::enable_if_t<support_crt_v<T>>*>
void FillKernel(Tensor& tensor, T val) {
  T* dptr = tensor.TypedPtr<T>();
  constexpr size_t unroll = sizeof(T) >= 4 ? 2 : 4;
  Device::SetCurrentDevice(tensor.GetDevice());
  FillKernelImpl<128, unroll>(tensor.NumElem(), dptr, val);
}

template void FillKernel<uint8_t, nullptr>(Tensor& t, uint8_t val);
template void FillKernel<uint16_t, nullptr>(Tensor& t, uint16_t val);
template void FillKernel<uint32_t, nullptr>(Tensor& t, uint32_t val);
template void FillKernel<uint64_t, nullptr>(Tensor& t, uint64_t val);

void CastCopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  CHECK_EQ(src.GetDevice().type, DeviceType::kCUDA);
  Device::SetCurrentDevice(src.GetDevice());

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

template <typename T>
void RandomUniformKernelImpl(Tensor& tensor, T low, T high) {
  T* data = tensor.TypedPtr<T>();
  size_t num_elem = tensor.NumElem();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;
  curandGenerateUniform(gen, data, num_elem);

  T scale = high - low;
  T bias = low;
  
  TensorIterator iter;
  iter.AddInput(tensor);
  iter.AddOutput(tensor);
  iter.Build();

  CUDAContiguousKernel(
    iter,
    [=] CUDA_LAMBDA (T elem) { return scale * elem + bias; });
}

template <>
void RandomUniformKernelImpl<double>(Tensor& tensor, double low, double high) {
  double* data = tensor.TypedPtr<double>();
  size_t num_elem = tensor.NumElem();
  auto gen = CUDAThreadLocalHandles::ThreadLocal().curand_gen;
  curandGenerateUniformDouble(gen, data, num_elem);

  double scale = high - low;
  double bias = low;
  
  TensorIterator iter;
  iter.AddInput(tensor);
  iter.AddOutput(tensor);
  iter.Build();

  CUDAContiguousKernel(
    iter,
    [=] CUDA_LAMBDA (double elem) { return scale * elem + bias; });
}

void RandomUniformKernel(Tensor& tensor, Scalar low, Scalar high) {
  CHECK_EQ(tensor.GetDevice().type, DeviceType::kCUDA);
  Device::SetCurrentDevice(tensor.GetDevice());

  if (tensor.GetDataType() == DataType::kHalf) {
    Tensor single_tensor = Tensor::SameAs(tensor, false, tensor.GetDevice(), DataType::kFloat);
    RandomUniformKernel(single_tensor, low, high);
    CastCopyKernel(single_tensor, tensor);
  } else {
    DTYPE_SWITCH_FLOAT_WITHOUT_HALF(tensor.GetDataType(), [&](){
      RandomUniformKernelImpl<scalar_t>(
        tensor, static_cast<scalar_t>(low), static_cast<scalar_t>(high));
    });
  }
}

} // namespace cuda
} // namespace ops
} // namespace tensor

#include <cstring>
#include "tensor/cpu_op.h"
#include "utils/random.h"

namespace tensor {
namespace ops {
namespace cpu {

void CopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice()); // use transfer for inter-device copy
  CHECK_EQ(src.GetDevice().type, DeviceType::kCPU);

  TensorIterator iter;
  iter.AddInput(src);
  iter.AddOutput(dst);
  iter.Build();

  DTYPE_SWITCH(src.GetDataType(), [&](){CPUElemwiseKernel(iter, [=](scalar_t elem) { return elem; }); });
}

void ElemwiseCopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice()); // use transfer for inter-device copy
  CHECK_EQ(src.GetDevice().type, DeviceType::kCPU);

  size_t size_in_bytes = src.TrueSizeInBytes();
  memcpy(dst.RawPtr(), src.RawPtr(), size_in_bytes);
}

void CastCopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDevice(), dst.GetDevice());
  CHECK_EQ(src.GetDevice().type, DeviceType::kCPU);

  TensorIterator iter;
  iter.AddInput(src);
  iter.AddOutput(dst);
  iter.Build();

  // this is an amazing double dtype dispatch...
  DTYPE_SWITCH(dst.GetDataType(), [&](){
    using dst_t = scalar_t;
    DTYPE_SWITCH(src.GetDataType(), [&](){
      CPUContiguousKernel(iter, [=](scalar_t elem) -> dst_t { return dtype_cast<scalar_t, dst_t, DeviceType::kCPU>::cast(elem); });
    });
  });
}

template <typename T, typename std::enable_if_t<support_crt_v<T>>* = nullptr>
void RandomUniformKernelImpl(Tensor& tensor, T low, T high) {
  T* dptr = tensor.TypedPtr<T>();
  size_t num_elem = tensor.NumElem();
  auto& generator = utils::RandomEngine::ThreadLocal();

  for (size_t i = 0; i < num_elem; ++i) {
    dptr[i] = generator.Uniform<T>(low, high);
  }
}

template <>
void RandomUniformKernelImpl<fp16_t>(Tensor& tensor, fp16_t low, fp16_t high) {
  fp16_t* dptr = tensor.TypedPtr<fp16_t>();
  size_t num_elem = tensor.NumElem();
  auto& generator = utils::RandomEngine::ThreadLocal();

  for (size_t i = 0; i < num_elem; ++i) {
    dptr[i] = static_cast<fp16_t>(generator.Uniform<float>(static_cast<float>(low), static_cast<float>(high)));
  }
}

template <typename T, typename std::enable_if_t<support_crt_v<T>>* = nullptr>
void RandomNormalKernelImpl(Tensor& tensor, T mean, T stddev) {
  T* dptr = tensor.TypedPtr<T>();
  size_t num_elem = tensor.NumElem();
  auto& generator = utils::RandomEngine::ThreadLocal();

  for (size_t i = 0; i < num_elem; ++i) {
    dptr[i] = generator.Normal<T>(mean, stddev);
  }
}

template <>
void RandomNormalKernelImpl<fp16_t>(Tensor& tensor, fp16_t mean, fp16_t stddev) {
  fp16_t* dptr = tensor.TypedPtr<fp16_t>();
  size_t num_elem = tensor.NumElem();
  auto& generator = utils::RandomEngine::ThreadLocal();

  for (size_t i = 0; i < num_elem; ++i) {
    dptr[i] = static_cast<fp16_t>(generator.Normal<float>(static_cast<float>(mean), static_cast<float>(stddev)));
  }
}

void RandomUniformKernel(Tensor& tensor, Scalar low, Scalar high) {
  CHECK_EQ(tensor.GetDevice().type, DeviceType::kCPU);
  DTYPE_SWITCH_FLOAT(tensor.GetDataType(), [&](){
    RandomUniformKernelImpl<scalar_t>(tensor, static_cast<scalar_t>(low), static_cast<scalar_t>(high));
  })
}

void RandomNormalKernel(Tensor& tensor, Scalar mean, Scalar stddev) {
  CHECK_EQ(tensor.GetDevice().type, DeviceType::kCPU);
  DTYPE_SWITCH_FLOAT(tensor.GetDataType(), [&](){
    RandomNormalKernelImpl<scalar_t>(tensor, static_cast<scalar_t>(mean), static_cast<scalar_t>(stddev));
  })
}

} // namespace cpu
} // namespace ops
} // namespace tensor


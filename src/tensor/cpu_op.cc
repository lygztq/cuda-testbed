#include <cstring>
#include "tensor/cpu_op.h"

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

  DTYPE_SWITCH(src.GetDataType(), CPUKernel(iter, [&](scalar_t elem) { return elem; }));
}

void ElemwiseCopyKernel(const Tensor& src, Tensor& dst) {
  CHECK_EQ(src.GetDataType(), dst.GetDataType());
  CHECK_EQ(src.GetDevice(), dst.GetDevice()); // use transfer for inter-device copy
  CHECK_EQ(src.GetDevice().type, DeviceType::kCPU);

  size_t size_in_bytes = src.TrueSizeInBytes();
  memcpy(dst.RawPtr(), src.RawPtr(), size_in_bytes);
}

Tensor ContiguousKernel(const Tensor& src) {
  Tensor contiguous = Tensor::SameAs(src, true, src.GetDevice());
  CopyKernel(src, contiguous);
  return contiguous;
}

} // namespace cpu
} // namespace ops
} // namespace tensor


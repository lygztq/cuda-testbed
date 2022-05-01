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

} // namespace cpu
} // namespace ops
} // namespace tensor


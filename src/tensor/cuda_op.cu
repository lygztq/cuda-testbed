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

} // namespace cuda
} // namespace ops
} // namespace tensor

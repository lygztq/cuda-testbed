#include <algorithm>
#include <functional>
#include <numeric>
#include "tensor/tensor.h"
#include "utils/logging.h"

namespace tensor {

std::shared_ptr<TensorStorage> TensorStorage::AllocStorage(
  size_t size, size_t align, Device device) {
  void* dptr = Device::AllocSpace(size, align, device);
  CHECK(dptr) << "Allocation space with size " 
              << size << " on "
              << GetDeviceName(device.type)
              << "(" << device.id << ")\n";

  return std::make_shared<TensorStorage>(dptr, size, align, device);
}

std::vector<size_t> Tensor::shape() const {
  std::vector<size_t> outShape(numAxis_, 0);
  std::copy_n(shape_.cbegin(), numAxis_, outShape.begin());
  return outShape;
}

std::vector<size_t> Tensor::stride() const {
  std::vector<size_t> outStride(numAxis_, 0);
  std::copy_n(stride_.cbegin(), numAxis_, outStride.begin());
  return outStride;
}

bool Tensor::IsContiguous() const {
  if (numAxis_ == 0) return true; // scalar or what?
  size_t s = 1;
  for (size_t i = numAxis_ - 1; i < numAxis_; --i) {
    if (stride_[i] == s) {
      s *= shape_[i];
    } else {
      return false;
    }
  }
  return true;
}

std::vector<size_t> Tensor::GenerateContiguousStride(std::vector<size_t> shape) {
  size_t numAxis = shape.size();
  std::vector<size_t> stride(numAxis, 1);
  for (size_t i = numAxis - 2; i < numAxis; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

// Tensor Tensor::View(std::vector<size_t> newShape) const {
//   size_t s1 = std::reduce(shape_.cbegin(), shape_.cend(), 1, std::multiplies<size_t>());
//   size_t s2 = std::reduce(newShape.cbegin(), newShape.cend(), 1, std::multiplies<size_t>());
//   CHECK(s1 == s2) << "Given new shape does not match the original shape\n";

//   // return Tensor()
// }

} // namespace tensor


#include <algorithm>
#include <functional>
#include <numeric>
#include "tensor/tensor.h"
#include "tensor/common_funcs.h"
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

TensorShapeInfo::TensorShapeInfo(const std::vector<size_t>& shape,
                                 const std::vector<size_t>& stride)
  : num_axes_(shape.size()) {
  CHECK_LE(shape.size(), kMaxTensorAxis) << "Input shape dimension out of range\n";
  CHECK_EQ(shape.size(), stride.size()) << "The number of dimension of"
                                        << "shape and stride should be equal\n";
  
  std::copy_n(shape.begin(), num_axes_, shape_.begin());
  std::copy_n(stride.begin(), num_axes_, stride_.begin());
}

std::vector<size_t> TensorShapeInfo::Shape() const {
  std::vector<size_t> out_shape(num_axes_, 0);
  std::copy_n(shape_.cbegin(), num_axes_, out_shape.begin());
  return out_shape;
}

std::vector<size_t> TensorShapeInfo::Stride() const {
  std::vector<size_t> out_stride(num_axes_, 0);
  std::copy_n(stride_.cbegin(), num_axes_, out_stride.begin());
  return out_stride;
}

std::vector<size_t> TensorShapeInfo::StrideInBytes(
  size_t dtype_size, size_t alignment) const {
  std::vector<size_t> stride_in_elem(Stride());
  size_t size = (alignment == 0) ? dtype_size : alignment;
  std::for_each(stride_in_elem.begin(), stride_in_elem.end(), [&](auto& e) { e *= size; });
  return stride_in_elem;
}

std::vector<size_t> TensorShapeInfo::ShapeInBytes(
  size_t dtype_size, size_t alignment) const {
  std::vector<size_t> shape_in_elem(Shape());
  size_t size = (alignment == 0) ? dtype_size : alignment;
  std::for_each(shape_in_elem.begin(), shape_in_elem.end(), [&](auto& e) { e *= size; });
  return shape_in_elem;
}

bool TensorShapeInfo::IsContiguous() const {
  if (num_axes_ == 0) return true; // scalar or what?
  size_t s = 1;
  for (size_t i = num_axes_ - 1; i < num_axes_; --i) {
    if (stride_[i] == s) {
      s *= shape_[i];
    } else {
      return false;
    }
  }
  return true;
}

std::vector<size_t> TensorShapeInfo::GenerateContiguousStride(std::vector<size_t> shape) {
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

Tensor Tensor::Empty(const std::vector<size_t>& shape,
                     DataType dtype,
                     size_t alignment,
                     Device device) {
  // return RawEmpty(shape, DataTypeSize(dtype), alignment, device);
  size_t elem_size = DataTypeSize(dtype);
  size_t numel = common::ShapeNumElem(shape);
  if (alignment == 0) alignment = elem_size;
  auto dptr = TensorStorage::AllocStorage(numel * elem_size, alignment, device);
  return Tensor(dptr, shape, TensorShapeInfo::GenerateContiguousStride(shape), dtype);
}

} // namespace tensor


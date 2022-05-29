#include <algorithm>
#include <functional>
#include <numeric>
#include "tensor/tensor.h"
#include "tensor/common.h"
#include "utils/logging.h"
#include "tensor/cpu_op.h"
#include "tensor/cuda_op.h"

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

Tensor Tensor::View(const std::vector<int>& view_shape) const {
  std::vector<size_t> actual_view_shape = common::ShapeDeduction(NumElem(), view_shape);
  CHECK(IsContiguous()) << "Only contiguous tensor can create views\n";

  std::vector<size_t> view_stride = TensorShapeInfo::GenerateContiguousStride(actual_view_shape);
  return Tensor(this->storage_, actual_view_shape, view_stride, GetDataType());
}

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

Tensor Tensor::SameAs(const Tensor& other,
                      bool contiguous,
                      Device device,
                      std::optional<DataType> dtype_opt) {
  device = (device.IsEmpty()) ? other.GetDevice() : device;
  DataType dtype = dtype_opt.value_or(other.GetDataType());

  auto shape = other.Shape();
  auto dptr = TensorStorage::AllocStorage(
    other.NumElem() * DataTypeSize(dtype), other.Alignment(), device);

  if (contiguous)
    return Tensor(dptr, shape, TensorShapeInfo::GenerateContiguousStride(shape), dtype);
  else
    return Tensor(dptr, shape, other.Stride(), dtype);
}

Tensor Tensor::Transfer(Device device) const {
  CHECK(device.Valid()) << "Dst device must be a valid device";
  
  if (device == GetDevice()) return *this;
  Tensor new_tensor = Tensor::SameAs(*this, false, device);
  Device::Transfer(
    RawPtr(), GetDevice(), new_tensor.RawPtr(), new_tensor.GetDevice(), TrueSizeInBytes());

  return new_tensor;
}

Tensor Tensor::Contiguous() const {
  if (IsContiguous()) return *this;
  CHECK_NE(GetDevice().type, DeviceType::kEmpty);

  Tensor contiguous = Tensor::SameAs(*this, true, GetDevice());

  switch (GetDevice().type) {
    case DeviceType::kCPU:
      ops::cpu::CopyKernel(*this, contiguous);
      break;
    case DeviceType::kCUDA:
      ops::cuda::CopyKernel(*this, contiguous);
      break;
    default:
      LOG_ERROR << "unknown device type\n";
      break;
  }

  return contiguous;
}

Tensor Tensor::DeepCopy() const {
  Tensor copy = Tensor::SameAs(*this);
  Device::Transfer(
    RawPtr(), GetDevice(), copy.RawPtr(), GetDevice(), TrueSizeInBytes());
  return copy;
}

Tensor Tensor::Transpose_(size_t i, size_t j) {
  CHECK_LT(i, NumAxes());
  CHECK_LT(j, NumAxes());
  if (i == j) return *this;

  using std::swap;
  auto& shape = ShapeRef();
  auto& stride = StrideRef();
  swap(shape[i], shape[j]);
  swap(stride[i], stride[j]);
  return *this;
}

Tensor Tensor::Transpose(size_t i, size_t j) const {
  Tensor transposed(*this);
  return transposed.Transpose_(i, j);
}

Tensor Tensor::Transpose_(const std::vector<size_t>& perm) {
  CHECK_EQ(perm.size(), NumAxes());
  std::array<bool, kMaxTensorAxis> vis;
  vis.fill(false);

  // check this is a valid perm
  for (size_t d : perm) {
    CHECK_LE(d, NumAxes());
    CHECK(!vis[d]);
    vis[d] = true;
  }

  auto& shape = ShapeRef();
  auto& stride = StrideRef();
  std::array<size_t, kMaxTensorAxis> new_shape;
  std::array<size_t, kMaxTensorAxis> new_stride;
  for (size_t i = 0; i < NumAxes(); ++i) {
    new_shape[i] = shape[perm[i]];
    new_stride[i] = stride[perm[i]];
  }

  std::copy_n(new_shape.begin(), NumAxes(), shape.begin());
  std::copy_n(new_stride.begin(), NumAxes(), stride.begin());

  return *this;
}

Tensor Tensor::Transpose(const std::vector<size_t>& perm) const {
  Tensor transposed(*this);
  return transposed.Transpose_(perm);
}

template <typename T, typename std::enable_if_t<support_crt_v<T>>* = nullptr>
Tensor FillImpl(Tensor& t, T val) {
  switch (t.GetDevice().type) {
    case DeviceType::kCPU:
      ops::cpu::FillKernel<T>(t, val);
      break;
    case DeviceType::kCUDA:
      ops::cuda::FillKernel<T>(t, val);
      break;
    default:
      LOG_ERROR << "unknown device type\n";
      break;
  }
  return t;
}

Tensor Tensor::FillInBytes(Tensor& t, void* val, size_t num_bytes) {
  switch (num_bytes) {
    case 1:
      FillImpl<uint8_t>(t, *(reinterpret_cast<uint8_t*>(val)));
      break;
    case 2:
      FillImpl<uint16_t>(t, *(reinterpret_cast<uint16_t*>(val)));
      break;
    case 4:
      FillImpl<uint32_t>(t, *(reinterpret_cast<uint32_t*>(val)));
      break;
    case 8:
      FillImpl<uint64_t>(t, *(reinterpret_cast<uint64_t*>(val)));
      break;
    default:
      LOG_ERROR << "Unsupported data type size\n";
      break;
  }
  return t;
}

Tensor Tensor::Cast(DataType dtype) const {
  if (dtype == GetDataType()) return *this;

  Tensor cast_res = Tensor::SameAs(*this, false, GetDevice(), dtype);
  switch (GetDevice().type) {
    case DeviceType::kCPU:
      ops::cpu::CastCopyKernel(*this, cast_res);
      break;
    case DeviceType::kCUDA:
      ops::cuda::CastCopyKernel(*this, cast_res);
      break;
    default:
      LOG_ERROR << "unknown device type\n";
      break;
  }
  return cast_res;
}

Tensor Tensor::Reshape(const std::vector<int>& new_shape) const {
  if (IsContiguous()) return this->View(new_shape);
  else return this->Contiguous().View(new_shape);
}

Tensor Tensor::Uniform(const std::vector<size_t>& shape,
                       Scalar low,
                       Scalar high,
                       DataType dtype,
                       Device device) {
  CHECK(dtype == DataType::kDouble ||
        dtype == DataType::kFloat ||
        dtype == DataType::kHalf) << "Only floating point type is supported.\n";
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  switch (device.type) {
    case DeviceType::kCPU:
      ops::cpu::RandomUniformKernel(new_tensor, low, high);
      break;
    case DeviceType::kCUDA:
      ops::cuda::RandomUniformKernel(new_tensor, low, high);
      break;
    default:
      LOG_ERROR << "unknown device type\n";
      break;
  }
  return new_tensor; 
}

Tensor Tensor::Normal(const std::vector<size_t>& shape,
                      Scalar mean,
                      Scalar stddev,
                      DataType dtype,
                      Device device) {
  CHECK(dtype == DataType::kDouble ||
        dtype == DataType::kFloat ||
        dtype == DataType::kHalf) << "Only floating point type is supported.\n";
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  switch (device.type) {
    case DeviceType::kCPU:
      ops::cpu::RandomNormalKernel(new_tensor, mean, stddev);
      break;
    case DeviceType::kCUDA:
      ops::cuda::RandomNormalKernel(new_tensor, mean, stddev);
      break;
    default:
      LOG_ERROR << "unknown device type\n";
      break;
  }
  return new_tensor; 
}

Tensor Tensor::Ones(const std::vector<size_t>& shape,
                    DataType dtype,
                    Device device) {
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  DTYPE_SWITCH(dtype, [&](){
    new_tensor.Fill(static_cast<scalar_t>(1));
  });
  return new_tensor;
}

Tensor Tensor::Zeros(const std::vector<size_t>& shape,
                     DataType dtype,
                     Device device) {
  Tensor new_tensor = Tensor::Empty(shape, dtype, 0, device);
  DTYPE_SWITCH(dtype, [&](){
    new_tensor.Fill(static_cast<scalar_t>(0));
  });
  return new_tensor;
}

} // namespace tensor


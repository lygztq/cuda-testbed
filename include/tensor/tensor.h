#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
#include "tensor/device.h"
#include "tensor/macros.h"
#include "tensor/dtype.h"
#include "tensor/common_funcs.h"

namespace tensor {

// static Device DefaultDevice_{DeviceType::kCPU, 0};

// forward decl
class Tensor;
class TensorRef;

constexpr size_t kMaxTensorAxis = 16;

class TensorStorage final {
public:
  explicit TensorStorage(void* data, size_t size, size_t align, Device device)
    : data_(data), size_(size), align_(align), device_(device) {}
  ~TensorStorage() { Device::FreeSpace(data_, device_); }

  // move and swap
  TensorStorage(TensorStorage&& other)
  : data_(other.data_), size_(other.size_)
  , align_(other.align_), device_(other.device_) {
    other.data_ = nullptr;
  }
  friend void swap(TensorStorage& t1, TensorStorage& t2) {
    using std::swap;
    swap(t1.data_, t2.data_);
    swap(t1.size_, t2.size_);
    swap(t1.align_, t2.align_);
    swap(t1.device_, t2.device_);
  }
  TensorStorage& operator=(TensorStorage&& other) {
    // move and swap
    using std::swap;
    TensorStorage tmp(std::move(other));
    swap(tmp, *this);
    return *this;
  }

  // no copy is allowed
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;

  // argument getters
  template <typename T>
  T* TypedPtr() { return static_cast<T*>(data_); }
  template <typename T>
  const T* TypedPtr() const { return static_cast<const T*>(data_); }

  void* RawPtr() { return data_; }
  const void* RawPtr() const { return data_; }

  size_t GetSize() const { return size_; }
  size_t GetAlignment() const { return align_; }
  Device GetDevice() const { return device_; }

  // allocate successfully or throw
  TENSOR_DLL static std::shared_ptr<TensorStorage> AllocStorage(size_t size, size_t align, Device device);

private:
  void* data_;
  size_t size_;
  size_t align_;
  Device device_;
};

class TensorShapeInfo final {
  friend Tensor;
  friend TensorRef;

public:
  TensorShapeInfo() = default;
  TENSOR_DLL explicit TensorShapeInfo(
    const std::vector<size_t>& shape, const std::vector<size_t>& stride);
  
  // argument getters
  size_t NumAxes() const { return num_axes_; }
  size_t Shape(size_t i) const { CHECK_LT(i, num_axes_); return shape_[i]; }
  TENSOR_DLL std::vector<size_t> Shape() const;
  const std::array<size_t, kMaxTensorAxis>& ShapeRef() const { return shape_; }
  std::array<size_t, kMaxTensorAxis>& ShapeRef() { return shape_; }
  size_t Stride(size_t i) const { CHECK_LT(i, num_axes_); return stride_[i]; }
  TENSOR_DLL std::vector<size_t> Stride() const;
  const std::array<size_t, kMaxTensorAxis>& StrideRef() const { return stride_; }
  std::array<size_t, kMaxTensorAxis>& StrideRef() { return stride_; }
  
  TENSOR_DLL std::vector<size_t> StrideInBytes(size_t dtype_size, size_t alignment = 0) const;
  TENSOR_DLL std::vector<size_t> ShapeInBytes(size_t dtype_size, size_t alignment = 0) const;

  TENSOR_DLL bool IsContiguous() const;
  TENSOR_DLL static std::vector<size_t> GenerateContiguousStride(std::vector<size_t> shape);
  void ChangeNumAxis(size_t n) {
    CHECK_LE(n, kMaxTensorAxis);
    num_axes_ = n;
  }

private:
  size_t num_axes_;
  std::array<size_t, kMaxTensorAxis> shape_;
  std::array<size_t, kMaxTensorAxis> stride_;
};

#define DECLARE_SHAPE_INFO \
private: \
  TensorShapeInfo shape_info_;

#define DECLARE_SHAPE_INFO_FUNCS \
public: \
size_t NumAxes() const { return shape_info_.NumAxes(); } \
size_t Shape(size_t i) const { return shape_info_.Shape(i); } \
std::vector<size_t> Shape() const { return shape_info_.Shape(); } \
const std::array<size_t, kMaxTensorAxis>& ShapeRef() const { return shape_info_.ShapeRef(); } \
std::array<size_t, kMaxTensorAxis>& ShapeRef() { return shape_info_.ShapeRef(); } \
size_t Stride(size_t i) const { return shape_info_.Stride(i); } \
std::vector<size_t> Stride() const { return shape_info_.Stride(); } \
const std::array<size_t, kMaxTensorAxis>& StrideRef() const { return shape_info_.StrideRef(); } \
std::array<size_t, kMaxTensorAxis>& StrideRef() { return shape_info_.StrideRef(); } \
bool IsContiguous() const { return shape_info_.IsContiguous(); } \
const TensorShapeInfo& GetShapeInfo() const { return shape_info_; } \
TensorShapeInfo& GetShapeInfo() { return shape_info_; } \
size_t NumElem() const { \
    return common::ShapeNumElem(shape_info_.shape_, shape_info_.num_axes_); \
} \

#define HAVE_SHAPE_INFO DECLARE_SHAPE_INFO DECLARE_SHAPE_INFO_FUNCS

template <typename T>
void ElemCountToByteCount(std::vector<size_t>& elem_shape) {
  std::for_each(elem_shape.begin(), elem_shape.end(), [](size_t &c) { c *= sizeof(T); });
}

#ifdef _WIN32
// template class TENSOR_DLL std::shared_ptr<TensorStorage>;
#endif // _WIN32
class Tensor final {
  friend TensorRef;
  HAVE_SHAPE_INFO
public:
  explicit Tensor(std::shared_ptr<TensorStorage> storage,
                  const std::vector<size_t>& shape,
                  const std::vector<size_t>& stride,
                  DataType dtype)
    : storage_(storage), shape_info_(shape, stride), dtype_(dtype) {}

  Tensor(const Tensor &) = default;
  
  Tensor(Tensor && other)
    : storage_(std::move(other.storage_))
    , shape_info_(std::move(other.shape_info_))
    , dtype_(other.dtype_) {}
  
  friend void swap(Tensor& t1, Tensor& t2) {
    using std::swap;
    swap(t1.storage_, t2.storage_);
    swap(t1.shape_info_, t2.shape_info_);
    swap(t1.dtype_, t2.dtype_);
  }
  
  Tensor& operator=(const Tensor& other) {
    using std::swap;
    auto tmp(other);
    swap(tmp, *this);
    return *this;
  }
  
  Tensor& operator=(Tensor&& other) {
    using std::swap;
    auto tmp(std::move(other));
    swap(tmp, *this);
    return *this;
  }

  // argument getters
  Device GetDevice() const { return storage_->GetDevice(); }
  size_t Alignment() const { return storage_->GetAlignment(); }
  size_t TrueSizeInBytes() const { return storage_->GetSize(); }
  DataType GetDataType() const { return dtype_; }
  size_t ElemSize() const { return DataTypeSize(dtype_); }

  template <typename T>
  T* TypedPtr() { return storage_->typename TypedPtr<T>(); }
  template <typename T>
  const T* TypedPtr() const { return storage_->typename TypedPtr<T>(); }
  void* RawPtr() { return storage_->RawPtr(); }
  const void* RawPtr() const { return storage_->RawPtr(); }

  // funcs
  // /* [TODO] */ Tensor Contiguous() const;
  // /* [TODO] */ Tensor DeepCopy() const;
  
  // Returns a new tensor with the same data as the self tensor but of a different shape.
  // /* [TODO] */ Tensor View(std::vector<size_t> newShape) const;

  // tensor creater

  TENSOR_DLL static Tensor Empty(const std::vector<size_t>& shape,
                                 DataType dtype,
                                 size_t alignment = 0,
                                 Device device = Device::DefaultDevice());

  // create a new tensor with same shape/dtype/device info as another tensor
  // Note: the new tensor is contiguous is arg contiguous is true
  TENSOR_DLL static Tensor SameAs(const Tensor& other, bool contiguous=false, Device device = Device::EmptyDevice());

  // transfer data of a tensor to another device, return a new tensor
  TENSOR_DLL Tensor Transfer(Device device) const;

  // template <typename T>
  // /* [TODO] */ TENSOR_DLL static Tensor Full(const std::vector<size_t>& shape, T val, size_t alignment = 0, Device device = Device::DefaultDevice());

private:
  // /* [TODO] */ void CopyFromTo();
  std::shared_ptr<TensorStorage> storage_;
  DataType dtype_;
};

// This reference is valid when the referred Tensor object is alive.
// But this is not safe if the referred object is dead.
#ifdef _WIN32
template class TENSOR_DLL std::weak_ptr<TensorStorage>;
#endif // _WIN32
class TENSOR_DLL TensorRef final {
  HAVE_SHAPE_INFO
public:
  TensorRef() = default;
  explicit TensorRef(const Tensor& t)
    : shape_info_(t.shape_info_)
    , storage_ref_(t.storage_)
    , dtype_(t.dtype_) {}

  TensorRef& operator=(const Tensor& t) {
    shape_info_ = t.shape_info_;
    storage_ref_ = t.storage_;
    dtype_ = t.dtype_;
    return *this;
  }

  Device GetDevice() const { return getStorage()->GetDevice(); }
  size_t Alignment() const { return getStorage()->GetAlignment(); }
  size_t TrueSizeInBytes() const { return getStorage()->GetSize(); }
  DataType GetDataType() const { return dtype_; }
  size_t ElemSize() const { return DataTypeSize(dtype_); }

  template <typename T>
  T* TypedPtr() { return getStorage()->typename TypedPtr<T>(); }
  template <typename T>
  const T* TypedPtr() const { return getStorage()->typename TypedPtr<T>(); }
  void* RawPtr() { return getStorage()->RawPtr(); }
  const void* RawPtr() const { return getStorage()->RawPtr(); }

private:
  // this function should only be used inside this class.
  // no shared pointer shall be leaked from this class.
  std::shared_ptr<TensorStorage> getStorage() const {
    if (auto s = storage_ref_.lock()) {
      return s;
    } else {
      return nullptr;
    }
  }

  std::weak_ptr<TensorStorage> storage_ref_;
  DataType dtype_;
};
  
} // namespace tensor

#endif // TENSOR_H_

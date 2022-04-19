#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
#include "tensor/device.h"
#include "tensor/macros.h"

namespace tensor {

// forward decl
class TENSOR_DLL Tensor;
class TENSOR_DLL TensorRef;

constexpr size_t kMaxTensorAxis = 16;

class TENSOR_DLL TensorStorage final {
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

  size_t size() const { return size_; }
  size_t alignment() const { return align_; }
  Device device() const { return device_; }

  // allocate successfully or throw
  static std::shared_ptr<TensorStorage> AllocStorage(size_t size, size_t align, Device device);

private:
  void* data_;
  size_t size_;
  size_t align_;
  Device device_;
};

class TENSOR_DLL TensorShapeInfo final {
  friend Tensor;
  friend TensorRef;

public:
  TensorShapeInfo() = default;
  explicit TensorShapeInfo(
    const std::vector<size_t>& shape, const std::vector<size_t>& stride);
  
  // argument getters
  size_t num_axis() const { return numAxis_; }
  size_t shape(size_t i) const { return shape_[i]; }
  std::vector<size_t> shape() const;
  const std::array<size_t, kMaxTensorAxis>& shape_const_ref() const { return shape_; }
  size_t stride(size_t i) const { return stride_[i]; }
  std::vector<size_t> stride() const;
  const std::array<size_t, kMaxTensorAxis>& stride_const_ref() const { return stride_; }

  bool IsContiguous() const;
  static std::vector<size_t> GenerateContiguousStride(std::vector<size_t> shape);

private:
  size_t numAxis_;
  std::array<size_t, kMaxTensorAxis> shape_;
  std::array<size_t, kMaxTensorAxis> stride_;
};

#define DECLARE_SHAPE_INFO \
private: \
  TensorShapeInfo shape_info_;

#define DECLARE_SHAPE_INFO_FUNCS \
public: \
size_t num_axis() const { return shape_info_.num_axis(); } \
size_t shape(size_t i) const { return shape_info_.shape(i); } \
std::vector<size_t> shape() const { return shape_info_.shape(); } \
const std::array<size_t, kMaxTensorAxis>& shape_const_ref() const { return shape_info_.shape_const_ref(); } \
size_t stride(size_t i) const { return shape_info_.stride(i); } \
std::vector<size_t> stride() const { return shape_info_.stride(); } \
const std::array<size_t, kMaxTensorAxis>& stride_const_ref() const { return shape_info_.stride_const_ref(); } \
bool IsContiguous() const { return shape_info_.IsContiguous(); } \
const TensorShapeInfo& shape_info() const { return shape_info_; } \
TensorShapeInfo& shape_info() { return shape_info_; }

#define HAVE_SHAPE_INFO DECLARE_SHAPE_INFO DECLARE_SHAPE_INFO_FUNCS

class TENSOR_DLL Tensor final {
  friend TensorRef;
  HAVE_SHAPE_INFO
public:
  explicit Tensor(std::shared_ptr<TensorStorage> storage,
                  const std::vector<size_t>& shape,
                  const std::vector<size_t>& stride)
    : storage_(storage), shape_info_(shape, stride) {}

  Tensor(const Tensor &) = default;
  
  Tensor(Tensor && other)
    : storage_(std::move(other.storage_))
    , shape_info_(std::move(other.shape_info_)) {}
  
  friend void swap(Tensor& t1, Tensor& t2) {
    using std::swap;
    swap(t1.storage_, t2.storage_);
    swap(t1.shape_info_, t2.shape_info_);
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
  Device device() const { return storage_->device(); }
  size_t alignment() const { return storage_->alignment(); }
  size_t trueSize() const { return storage_->size(); }

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
  // /* [TODO] */ static Tensor Empty(std::vector<size_t> shape, size_t dtypeSize, size_t alignment = 0);
  // template <typename T>
  // /* [TODO] */ static Tensor Full(std::vector<size_t> shape, T val, size_t alignment = 0);

private:
  // /* [TODO] */ void CopyFromTo();
  std::shared_ptr<TensorStorage> storage_;
};


class TENSOR_DLL TensorRef final {
  HAVE_SHAPE_INFO
public:
  explicit TensorRef(const Tensor& t)
    : shape_info_(t.shape_info_)
    , storage_ref_(t.storage_) {}

  TensorRef& operator=(const Tensor& t) {
    shape_info_ = t.shape_info_;
    storage_ref_ = t.storage_;
    return *this;
  }

  Device device() const { return getStorage()->device(); }
  size_t alignment() const { return getStorage()->alignment(); }
  size_t trueSize() const { return getStorage()->size(); }

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
};
  
} // namespace tensor

#endif // TENSOR_H_

#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
#include "tensor/device.h"
#include "tensor/macros.h"

namespace tensor {

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

class TENSOR_DLL Tensor final {
public:
  explicit Tensor(std::shared_ptr<TensorStorage> storage,
                  std::vector<size_t> shape,
                  std::vector<size_t> stride);

  Tensor(const Tensor &) = default;
  
  Tensor(Tensor && other)
    : storage_(std::move(other.storage_))
    , numAxis_(other.numAxis_)
    , shape_(std::move(other.shape_))
    , stride_(std::move(other.stride_)) {}
  
  friend void swap(Tensor& t1, Tensor& t2) {
    using std::swap;
    swap(t1.storage_, t2.storage_);
    swap(t1.shape_, t2.shape_);
    swap(t1.stride_, t2.stride_);
    swap(t1.numAxis_, t2.numAxis_);
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
  size_t numAxis() const { return numAxis_; }
  size_t shape(size_t i) const { return shape_[i]; }
  std::vector<size_t> shape() const;
  size_t stride(size_t i) const { return stride_[i]; }
  std::vector<size_t> stride() const;
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
  bool IsContiguous() const;
  // /* [TODO] */ Tensor Contiguous() const;
  static std::vector<size_t> GenerateContiguousStride(std::vector<size_t> shape);
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
  size_t numAxis_;
  std::array<size_t, kMaxTensorAxis> shape_;
  std::array<size_t, kMaxTensorAxis> stride_;
};
  
} // namespace tensor


#endif // TENSOR_H_

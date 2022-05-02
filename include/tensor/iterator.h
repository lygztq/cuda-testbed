/*!
 * \file iterator.h
 * \brief Tensor iterator and helper classes.
 */
#ifndef TENSOR_ITERATOR_H_
#define TENSOR_ITERATOR_H_

#include <vector>
#include <array>
#include <numeric>
#include "tensor/tensor.h"
#include "tensor/function_ref.h"
#include "tensor/macros.h"
#include "tensor/common.h"
#include "utils/logging.h"

namespace tensor {

struct Range {
  size_t start;
  size_t end;
  Range() = default;
  Range(size_t s, size_t e) : start(s), end(e) {}
};

class IndexCounter {
public:
  IndexCounter() = default;
  IndexCounter(const std::vector<size_t>& shape)
    : shape_(shape)
    , index_(std::vector<size_t>(shape.size(), 0)) {}

  bool IsFinish() const {
    if (overflow_) return true;
    bool finish = true;
    for (size_t i = 0; i < NumAxes(); ++i) {
      finish &= (index_[i] >= shape_[i]);
    }
    return finish;
  }

  void Advance(size_t axis_idx, size_t step = 1) {
    CHECK(axis_idx < NumAxes());
    size_t carry = step;
    for (size_t i = axis_idx; i <= axis_idx; --i) {
      index_[i] += carry;
      carry = index_[i] / shape_[i];
      index_[i] %= shape_[i];
    }
    if (carry) overflow_ = true;
  }

  void Reset() {
    memset(index_.data(), 0, sizeof(size_t) * index_.size());
    overflow_ = false;
  }

  // Accessors and mutators
  const std::vector<size_t>& Index() const { return index_; }
  const std::vector<size_t>& Shape() const { return shape_; }
  size_t NumAxes() const { return shape_.size(); }

private:
  std::vector<size_t> shape_;
  std::vector<size_t> index_;
  bool overflow_;
};

// https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/fcb4a74e-c6d4-4a20-94f9-1ad1669b429d/what-does-warning-c4251-class-needs-to-have-dll-interface-to-be-used-by-clients-of-class-mean
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4251)
#endif // _WIN32

/*!
 * \brief TensorIterator is a helper class for element-wise operations, such as
 *        arithmetic, comparisons, and trigonometric functions. It handles
 *        broadcasting and type conversions of operands.
 *
 *        This is inspired by NumPy's Array Iterator API (NpyIter) and TensorIterator
 *        in Pytorch.
 *
 *        The files Loops.h provide functions to build kernels that use TensorIterator.
 */
class TensorIterator {
  using loop2d_t = function_ref<void(char**, const size_t*, size_t, size_t)>;

public:
  TensorIterator() = default;

  TENSOR_DLL void ForEach(loop2d_t loop);

  TENSOR_DLL void FixTensors();

  TENSOR_DLL void BroadcastShape();

  TENSOR_DLL void InitShape();

  TENSOR_DLL void CompressShape();

  TENSOR_DLL void Build();

  void AddInput(const Tensor& t) { in_tensors_.emplace_back(t); }

  void AddOutput(const Tensor& t) { out_tensors_.emplace_back(t); }

  size_t NumTensors() const {
    return NumInTensors() + NumOutTensors();
  }
  size_t NumInTensors() const {
    return has_fixed_tensor_ ? num_in_tensors_ : in_tensors_.size();
  }
  size_t NumOutTensors() const {
    return has_fixed_tensor_ ? num_out_tensors_ : out_tensors_.size();
  }

  // attributes
  const std::vector<size_t>& Shape() const { return shape_; }

  std::vector<size_t>& Shape() { return shape_; }

  const std::vector<TensorRef>& Tensors() const { return tensors_; }

  std::vector<TensorRef>& Tensors() { return tensors_; }

  size_t NumAxes() const { return shape_.size(); }

  size_t NumElem() const {
    return common::ShapeNumElem(shape_);
  }

  bool HasInit() const { return has_init_; }
  bool HasFixedTensor() const { return has_fixed_tensor_; }
  bool HasBroadCastedShape() const { return has_broadcasted_shape_; }
  bool Valid() const { return has_init_ && has_fixed_tensor_ && has_broadcasted_shape_; }
  bool HasCompressedShape() const { return has_compressed_; }

private:
  std::vector<size_t> getStridesInBytes() const;
  void getDataPtrs(std::vector<char*>& dptrs,
                   const std::vector<char*>& base,
                   const std::vector<size_t>& index,
                   const std::vector<size_t>& stride_bytes) const;

  std::vector<TensorRef> in_tensors_;
  std::vector<TensorRef> out_tensors_;
  std::vector<TensorRef> tensors_;
  std::vector<size_t> shape_;
  size_t num_in_tensors_;
  size_t num_out_tensors_;

  bool has_fixed_tensor_ = false;
  bool has_init_ = false;
  bool has_broadcasted_shape_ = false;
  bool has_compressed_ = false;
};

#ifdef _WIN32
#pragma warning(pop)
#endif // _WIN32

} // namespace tensor

#endif  // TENSOR_ITERATOR_H_

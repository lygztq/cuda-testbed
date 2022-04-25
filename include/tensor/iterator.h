#ifndef TENSOR_ITERATOR_H_
#define TENSOR_ITERATOR_H_

#include <vector>
#include <array>
#include <numeric>
#include "tensor/tensor.h"
#include "tensor/function_ref.h"
#include "tensor/macros.h"
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

} // namespace tensor


#endif  // TENSOR_ITERATOR_H_

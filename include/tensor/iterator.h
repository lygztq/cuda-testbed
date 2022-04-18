#ifndef TENSOR_ITERATOR_H_
#define TENSOR_ITERATOR_H_

#include <vector>
#include <array>
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
  IndexCounter(std::vector<size_t> shape)
    : shape_(shape)
    , index_(std::vector<size_t>(shape.size(), 0)) {}

  bool IsFinish() const {
    bool finish = true;
    for (size_t i = 0; i < dim(); ++i) {
      finish &= (index_[i] >= shape_[i]);
    }
    return finish;
  }

  void Advance(size_t dimidx, size_t step = 1) {
    CHECK(dimidx < dim());
    size_t carry = step;
    for (size_t i = dimidx; i <= dimidx; --i) {
      index_[i] += carry;
      carry = index_[i] / shape_[i];
      index_[i] %= shape_[i];
    }
  }

  // Accessors and mutators
  const std::vector<size_t>& index() const { return index_; }
  const std::vector<size_t>& shape() const { return shape_; }
  size_t dim() const { return shape_.size(); }

protected:
  std::vector<size_t> shape_;
  std::vector<size_t> index_;
};

} // namespace tensor


#endif  // TENSOR_ITERATOR_H_

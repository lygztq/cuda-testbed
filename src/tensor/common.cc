#include "tensor/common.h"
#include "utils/logging.h"

namespace tensor {
namespace common {

std::vector<size_t> ShapeDeduction(size_t num_elem, const std::vector<int>& shape) {
  bool meet_minus_one = false;
  size_t prod = 1;
  size_t deduction_pos = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      CHECK(!meet_minus_one);
      meet_minus_one = true;
      deduction_pos = i;
      continue;
    } else if (shape[i] > 0) {
      prod *= shape[i];
    } else {
      LOG_ERROR << "Invalid shape\n";
    }
  }

  if (meet_minus_one) {
    CHECK_EQ((num_elem % prod), 0) << "Given shape is unmatched (un-divisible) with the given number of elements\n";
  } else {
    CHECK_EQ(prod, num_elem) << "Given shape is unmatched with the given number of elements\n";
  }

  std::vector<size_t> new_shape(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    new_shape[i] = (shape[i] == -1) ? (num_elem / prod) : shape[i];
  }
  return new_shape;
}

} // namespace common
} // namespace tensor


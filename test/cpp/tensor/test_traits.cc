#include <vector>
#include <numeric>
#include <gtest/gtest.h>
#include "tensor/traits.h"

namespace {

float func_a(int a, float b, char c) {
  return a + b + (int)c;
}

} // namespace


TEST(TestTraits, TestDeferenceTrait) {
  using trait = tensor::function_traits<decltype(func_a)>;

  // shape: (2, 3)
  std::vector<int> a_vec(6);
  std::vector<float> b_vec(6);
  std::vector<char> c_vec(6);
  std::iota(a_vec.begin(), a_vec.end(), 1);
  std::iota(b_vec.begin(), b_vec.end(), 1.f);
  std::iota(c_vec.begin(), c_vec.end(), 'a');

  // stride: (3, 1)
  std::vector<size_t> stride = {1, 1, 1};
  
  // data ptrs
  std::vector<char*> datas = {(char*)a_vec.data(), (char*)b_vec.data(), (char*)c_vec.data()};

  auto o = std::apply(func_a, tensor::deference<trait>(
    datas.data(), stride.data(), 0));
  EXPECT_EQ(o, 2.f + ((float)'a'));
}

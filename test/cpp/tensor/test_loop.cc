#include <vector>
#include <numeric>
#include <gtest/gtest.h>
#include "tensor/loop.h"

namespace {

std::vector<size_t> get_2d_stride(std::vector<std::vector<size_t>> v) {
  size_t ntensor = v.size();
  size_t ndim = v[0].size();
  std::vector<size_t> strides(2 * ntensor, 0);
  auto s_iter = strides.begin();
  for (size_t i = 1; i < 2; --i) {
    for (size_t t = 0; t < ntensor; ++t) {
      *s_iter = v[t][ndim - i - 1];
      ++s_iter;
    }
  }
  return strides;
}

std::vector<size_t> concat_strides(std::vector<std::vector<size_t>> v) {
  size_t ntensor = v.size();
  size_t ndim = v[0].size();
  std::vector<size_t> strides(ndim * ntensor, 0);
  auto s_iter = strides.begin();
  for (size_t i = 0; i < ndim; ++i) {
    for (size_t t = 0; t < ntensor; ++t) {
      *s_iter = v[t][i];
      ++s_iter;
    }
  }
  return strides;
}

}

// TODO: non-contiguous tensor test
TEST(TestLoop, TestLoop2d) {
  // shape: (2, 3, 2), numel = 12
  std::vector<int> a_vec(12);
  std::vector<float> b_vec(12);
  std::vector<size_t> c_vec(12);
  std::vector<float> o_vec(12);
  std::iota(a_vec.begin(), a_vec.end(), 1);
  std::iota(b_vec.begin(), b_vec.end(), 2.f);
  std::iota(c_vec.begin(), c_vec.end(), 0);
  std::vector<char*> datas = {
    (char*)o_vec.data(),
    (char*)a_vec.data(),
    (char*)b_vec.data(),
    (char*)c_vec.data()};

  // -- Contiguous Case --
  // get ground truth
  std::vector<float> t_o(12);
  std::vector<size_t> idx(12);
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(idx.begin(), idx.end(), [&](size_t i) {
    t_o[i] = a_vec[i] + b_vec[i] + c_vec[i]; });

  // stride: (6, 2, 1)
  // stride needed: (dim_n, ..., dim_1, dim_0)
  // where dim_i = (tensor_0_s_i, tensor_1_s_i, ...)
  // Current stride: {tensor_0_s, tensor_1_s, ...}
  std::vector<size_t> a_stride{6 * sizeof(int), 2 * sizeof(int), sizeof(int)};
  std::vector<size_t> b_stride{6 * sizeof(float), 2 * sizeof(float), sizeof(float)};
  std::vector<size_t> c_stride{6 * sizeof(size_t), 2 * sizeof(size_t), sizeof(size_t)};
  std::vector<size_t> o_stride{6 * sizeof(float), 2 * sizeof(float), sizeof(float)};
  std::vector<size_t> stride_2d = get_2d_stride({
    o_stride, a_stride, b_stride, c_stride});
  std::vector<size_t> concat_stride = concat_strides({
    o_stride, a_stride, b_stride, c_stride});
  
  tensor::Loop2d loop([](int a, float b, size_t c) -> float {return a + b + c; });
  loop(datas.data(), stride_2d.data(), 2, 3);
  for (size_t i = 0; i < datas.size(); ++i) {
    datas[i] += concat_stride[i];
  }
  loop(datas.data(), stride_2d.data(), 2, 3);
  
  for (size_t i = 0; i < o_vec.size(); ++i) {
    EXPECT_EQ(o_vec[i], t_o[i]);
  }

  // -- Incontiguous Case --
  std::swap(a_stride[0], a_stride[2]);
  std::swap(c_stride[0], c_stride[2]);
  a_vec = {1,7,3,9,5,11,2,8,4,10,6,12};
  c_vec = {0,6,2,8,4,10,1,7,3,9,5,11};
  stride_2d = get_2d_stride({
    o_stride, a_stride, b_stride, c_stride});
  concat_stride = concat_strides({
    o_stride, a_stride, b_stride, c_stride});
  datas = {
    (char*)o_vec.data(),
    (char*)a_vec.data(),
    (char*)b_vec.data(),
    (char*)c_vec.data()};
  loop(datas.data(), stride_2d.data(), 2, 3);
  for (size_t i = 0; i < datas.size(); ++i) {
    datas[i] += concat_stride[i];
  }
  loop(datas.data(), stride_2d.data(), 2, 3);
  
  for (size_t i = 0; i < o_vec.size(); ++i) {
    EXPECT_EQ(o_vec[i], t_o[i]);
  }
}

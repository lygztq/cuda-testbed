#include <vector>
#include <gtest/gtest.h>
#include "tensor/tensor.h"
#include "tensor/iterator.h"

// TODO: test incontiguous case
TEST(TestIterator, TestIteratorShapeInference) {
  using tensor::Tensor;
  using tensor::TensorIterator;

  Tensor i0 = Tensor::Empty({2,3,4}, sizeof(float));
  Tensor i1 = Tensor::Empty({1,2,1,4}, sizeof(double));
  Tensor o0 = Tensor::Empty({2,2,3,4}, sizeof(float));
  TensorIterator iter;
  iter.AddInput(i0);
  iter.AddInput(i1);
  iter.AddOutput(o0);

  iter.FixTensors();
  EXPECT_EQ(iter.NumInTensors(), 2);
  EXPECT_EQ(iter.NumOutTensors(), 1);

  // init shape
  iter.InitShape();
  EXPECT_EQ(iter.NumAxes(), 4);

  iter.BroadcastShape();
  std::vector<size_t> expect_shape{2,2,3,4};
  EXPECT_EQ(expect_shape.size(), iter.Shape().size());
  for (size_t i = 0; i < expect_shape.size(); ++i) {
    EXPECT_EQ(expect_shape[i], iter.Shape()[i]);
  }

  iter.CompressShape();
  expect_shape.resize(1);
  expect_shape[0] = 2 * 2 * 3 * 4;
  EXPECT_EQ(expect_shape.size(), iter.Shape().size());
  for (size_t i = 0; i < expect_shape.size(); ++i) {
    EXPECT_EQ(expect_shape[i], iter.Shape()[i]);
  }
}



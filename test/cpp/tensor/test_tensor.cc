#include <memory>
#include <gtest/gtest.h>
#include "tensor/tensor.h"
#include "tensor/device.h"
#include "tensor/dtype.h"

TEST(TestTensor, TestTensorStorageCPU) {
  tensor::Device d = tensor::Device(tensor::DeviceType::kCPU, 0);
  auto ptr = tensor::TensorStorage::AllocStorage(24 * sizeof(int), sizeof(int), d);
  EXPECT_EQ((ptr->GetAlignment()), sizeof(int));
  EXPECT_EQ((ptr->GetSize()), 24 * sizeof(int));
  EXPECT_NE((ptr->RawPtr()), nullptr);
  EXPECT_EQ((ptr->GetDevice()), d);
}

TEST(TestTensor, TestTensorShapeInfo) {
  using tensor::TensorShapeInfo;
  std::vector<size_t> shape{2,3,4};
  std::vector<size_t> stride = TensorShapeInfo::GenerateContiguousStride(shape);
  TensorShapeInfo shape_info(shape, stride);

  EXPECT_EQ(shape_info.NumAxes(), 3);
  EXPECT_EQ(shape_info.Shape(1), 3);
  EXPECT_EQ(shape_info.Stride(1), 4);
  EXPECT_ANY_THROW(shape_info.Shape(100));
  EXPECT_TRUE(shape_info.IsContiguous());
}

TEST(TestTensor, TestTensorCPU) {
  using tensor::Tensor;
  using tensor::Device;
  using tensor::DeviceType;
  using tensor::TensorRef;
  using tensor::DataType;

  Device d(DeviceType::kCPU, 0);
  Tensor t = Tensor::Empty({2,3,4}, DataType::kFloat);
  EXPECT_NE(t.RawPtr(), nullptr);
  TensorRef ref(t);
  EXPECT_NE(ref.RawPtr(), nullptr);
  EXPECT_EQ(ref.GetDevice(), d);
}



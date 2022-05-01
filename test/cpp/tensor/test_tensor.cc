#include <memory>
#include <gtest/gtest.h>
#include "tensor/tensor.h"
#include "tensor/device.h"
#include "tensor/dtype.h"

using tensor::Tensor;
using tensor::Device;
using tensor::DeviceType;
using tensor::DataType;

TEST(TestTensor, TestTensorStorageCPU) {
  Device d = Device(tensor::DeviceType::kCPU, 0);
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

TEST(TestTensor, TestTensorCreateCPU) {
  using tensor::TensorRef;

  Device d(DeviceType::kCPU, 0);
  Tensor t = Tensor::Empty({2,3,4}, DataType::kFloat);
  EXPECT_NE(t.RawPtr(), nullptr);
  TensorRef ref(t);
  EXPECT_NE(ref.RawPtr(), nullptr);
  EXPECT_EQ(ref.GetDevice(), d);
}

TEST(TestTensor, TestTensorCreateCUDA) {
  using tensor::TensorRef;

  Device d(DeviceType::kCUDA, 0);
  Tensor t = Tensor::Empty({2,3,4}, DataType::kFloat, 0, Device(DeviceType::kCUDA, 0));
  EXPECT_NE(t.RawPtr(), nullptr);
  TensorRef ref(t);
  EXPECT_NE(ref.RawPtr(), nullptr);
  EXPECT_EQ(ref.GetDevice(), d);
}

TEST(TestTensor, TestTransfer) {
  Tensor t = Tensor::Empty({2,3,4}, DataType::kFloat);
  float *raw_ptr_t = t.TypedPtr<float>();
  raw_ptr_t[0] = 1024.0;
  raw_ptr_t[6] = 512.0;
  Tensor t_cuda = t.Transfer(Device(DeviceType::kCUDA, 0));
  Tensor t_copy = t_cuda.Transfer(Device(DeviceType::kCPU, 0));

  float *raw_ptr_t_copy = t_copy.TypedPtr<float>();
  EXPECT_EQ(raw_ptr_t_copy[0], 1024.f);
  EXPECT_EQ(raw_ptr_t_copy[6], 512.f);
}

TEST(TestTensor, TestContiguous) {
  // cpu
  std::vector<size_t> shape{2,3,4};
  Tensor t = Tensor::Empty(shape, DataType::kFloat);
  EXPECT_TRUE(t.IsContiguous());
  t.Transpose_(0, 2);
  EXPECT_EQ(t.Shape(0), 4);
  EXPECT_EQ(t.Shape(2), 2);
  EXPECT_FALSE(t.IsContiguous());
  Tensor t_cont = t.Contiguous();
  EXPECT_EQ(t_cont.Shape(0), 4);
  EXPECT_EQ(t_cont.Shape(2), 2);
  EXPECT_TRUE(t_cont.IsContiguous());

  // cuda
  Tensor t_cuda = Tensor::Empty(shape, DataType::kFloat, 0, {DeviceType::kCUDA, 0});
  EXPECT_TRUE(t_cuda.IsContiguous());
  t_cuda.Transpose_(0, 2);
  EXPECT_EQ(t_cuda.Shape(0), 4);
  EXPECT_EQ(t_cuda.Shape(2), 2);
  EXPECT_FALSE(t_cuda.IsContiguous());
  Tensor t_cuda_cont = t_cuda.Contiguous();
  EXPECT_EQ(t_cuda_cont.Shape(0), 4);
  EXPECT_EQ(t_cuda_cont.Shape(2), 2);
  EXPECT_TRUE(t_cuda_cont.IsContiguous());
}

TEST(TestTensor, TestFull) {
  std::vector<size_t> shape{2,3,4};

  // cpu
  Tensor t_cpu = Tensor::Full(shape, 1.f);
  float* t_cpu_ptr = t_cpu.TypedPtr<float>();
  for (size_t i = 0; i < t_cpu.NumElem(); ++i) {
    EXPECT_EQ(t_cpu_ptr[i], 1.f);
  }

  // cuda
  Tensor t_cuda = Tensor::Full(shape, 3.14, 0, {DeviceType::kCUDA, 0});
  Tensor t_cuda_cpu = t_cuda.Transfer({DeviceType::kCPU, 0});
  double* t_cuda_cpu_ptr = t_cuda_cpu.TypedPtr<double>();
  for (size_t i = 0; i < t_cuda.NumElem(); ++i) {
    EXPECT_EQ(t_cuda_cpu_ptr[i], 3.14);
  }
}

TEST(TestTensor, TestTranspose) {
  
}

TEST(TestTensor, TestView) {

}

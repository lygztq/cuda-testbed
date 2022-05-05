#include "utils/logging.h"
#include "tensor/macros.h"
#include "tensor/tensor.h"
#include <cuda_runtime.h>
#include <iostream>

using tensor::Tensor;

__global__ void naive_matmul_kernel(float* RESTRICT a,
                                    float* RESTRICT b,
                                    float* RESTRICT c,
                                    int M,
                                    int K,
                                    int N) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int midx = gidx / N;
  int nidx = gidx % N;
  float res = 0.f;

  for (int i = 0; i < K; ++i) {
    res += a[midx * K + i] * b[i * K + nidx];
  }

  c[gidx] = res;
}

Tensor NaiveMatMul(Tensor t1, Tensor t2) {
  CHECK_EQ(t1.NumAxes(), 2);
  CHECK_EQ(t2.NumAxes(), 2);
  CHECK_EQ(t1.Shape(1), t2.Shape(0));
  CHECK_EQ(t1.GetDataType(), t2.GetDataType());
  CHECK_EQ(t1.GetDevice(), t2.GetDevice());
  CHECK_EQ(t1.GetDevice().type, tensor::DeviceType::kCUDA);

  Tensor ret = Tensor::Zeros({t1.Shape(0), t2.Shape(1)}, t1.GetDataType(), t1.GetDevice());
  int NT = 256;
  dim3 nt = NT;
  dim3 nb = ((int)ret.NumElem() + NT - 1) / NT;
  naive_matmul_kernel<<<nb, nt>>>(t1.TypedPtr<float>(), t2.TypedPtr<float>(), ret.TypedPtr<float>(), t1.Shape(0), t1.Shape(1), t2.Shape(1));
  return ret;
}

int main() {
  Tensor t1 = Tensor::Ones({1024, 512}, tensor::DataType::kFloat, {tensor::DeviceType::kCUDA, 0});
  Tensor t2 = Tensor::Ones({512, 1024}, tensor::DataType::kFloat, {tensor::DeviceType::kCUDA, 0});
  Tensor t3 = NaiveMatMul(t1, t2).Transfer({tensor::DeviceType::kCPU, 0});
  std::cout << t3.TypedPtr<float>()[0] << std::endl;
  return 0;
}

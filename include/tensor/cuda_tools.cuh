#ifndef TENSOR_CUDA_TOOLS_CUH_
#define TENSOR_CUDA_TOOLS_CUH_

#include <algorithm>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor/common.h"
#include "tensor/macros.h"

namespace tensor {
namespace ops {
namespace cuda {

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
TENSOR_HOST_DEVICE typename traits::return_t
invoke_impl(const func_t &f, char *const *data, const index_t *strides, int i,
            std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(*(typename traits::template arg<INDEX>::type*)(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
TENSOR_HOST_DEVICE typename traits::return_t
invoke(const func_t &f, char *const *data, const index_t *strides, int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

// template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
// TENSOR_HOST_DEVICE void
// foo(const func_t& f, char *const *data, const index_t *strides, int i) {}

template <size_t NARGS>
struct OffsetCalculator {
  OffsetCalculator(size_t num_axes,
                   const std::vector<size_t>& shape,
                   const std::vector<size_t>& strides,
                   const std::array<size_t, NARGS>& elem_size) 
    : num_axes_(num_axes) {
    CHECK_LE(num_axes, kMaxTensorAxis);

#ifndef _MSC_VER
    #pragma unroll
#endif
    for (size_t t = 0; t < NARGS; ++t) {
      elem_size_[t] = elem_size[t];
    }

    for (size_t i = 0; i < num_axes; ++i) {
      shape_[i] = shape[i];
#ifndef _MSC_VER
      #pragma unroll
#endif
      for (size_t t = 0; t < NARGS; ++t) {
        strides_[i][t] = strides[i * NARGS + t];
      }
    }
  }
  // offset must be a pointer points to a array with size kMaxTensorAxis
  TENSOR_HOST_DEVICE void get(size_t gidx, size_t* offset) const {
#ifndef _MSC_VER
    #pragma unroll
#endif
    for (size_t i = 0; i < kMaxTensorAxis; ++i) {
      offset[i] = 0;
    }

    size_t mod;
    for (size_t d = num_axes_ - 1; d < num_axes_; --d) {
      mod = gidx % shape_[d];
      gidx = gidx / shape_[d];

#ifndef _MSC_VER
      #pragma unroll
#endif
      for (size_t t = 0; t < NARGS; ++t) {
        offset[t] += mod * strides_[d][t] * elem_size_[t];
      }
    }
  }

  size_t num_axes_;
  size_t shape_[kMaxTensorAxis]; // std::array is an experimental feature in libcu++
  size_t strides_[kMaxTensorAxis][std::max<size_t>(NARGS, 1ULL)];
  size_t elem_size_[std::max<size_t>(NARGS, 1ULL)];
};

template<size_t nt, size_t vt, typename func_t>
__global__ void elementwise_kernel(size_t N, func_t f) {
  int tid = threadIdx.x;
  int nv = (int)(nt * vt);
  int idx = nv * blockIdx.x + tid;
#ifndef _MSC_VER
  #pragma unroll
#endif
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <size_t nt, size_t vt, typename T>
__global__ void fill_kernel(size_t N, T* dptr, T val) {
  int tid = threadIdx.x;
  int nv = (int)(nt * vt);
  int idx = nv * blockIdx.x + tid;
#ifndef _MSC_VER
  #pragma unroll
#endif
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      dptr[idx] = val;
      idx += nt;
    }
  }
}

template <size_t nt, size_t vt, typename Op>
void cudaElemwiseKernelImpl(size_t N, Op&& op) {
  if (N == 0) return;

  dim3 block(static_cast<unsigned int>(nt));
  dim3 grid(static_cast<unsigned int>((N + block.x * vt - 1) / (block.x * vt)));
  elementwise_kernel<nt, vt, Op><<<grid, block>>>(N, op);
  CUDA_CALL(cudaGetLastError());
}

template <size_t nt, size_t vt, typename T>
void FillKernelImpl(size_t N, T* dptr, T val) {
  if (N == 0) return;

  dim3 block(static_cast<unsigned int>(nt));
  dim3 grid(static_cast<unsigned int>((N + block.x * vt - 1) / (block.x * vt)));
  fill_kernel<nt, vt, T><<<grid, block>>>(N, dptr, val);
  CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace ops

// TENSOR_HOST_DEVICE: actually these are only __device__ function, but we need to call it in a
// __host__ __device__ lambda function. Why we use __host__ __device__ lambda? Because we need
// its function_traits on CPU.
template <typename T1, typename T2>
struct dtype_cast<T1, T2, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static T2 cast(T1 src) { return static_cast<T2>(src); }
};

template <typename T>
struct dtype_cast<T, fp16_t, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static fp16_t cast(T src) {
    __half temp = __float2half_rn(static_cast<float>(src));
    return *reinterpret_cast<fp16_t*>(&temp);
  }
};

template <typename T>
struct dtype_cast<fp16_t, T, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static T cast(fp16_t src) {
    float temp = __half2float(
      *reinterpret_cast<__half*>(&src));
    return static_cast<T>(temp);
  }
};

template <>
struct dtype_cast<fp16_t, fp16_t, DeviceType::kCUDA> {
  TENSOR_HOST_DEVICE static fp16_t cast(fp16_t src) {
    return src;
  }
};

} // namespace tensor

#endif  // TENSOR_CUDA_TOOLS_CUH_

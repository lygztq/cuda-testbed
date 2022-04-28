#include <cstring>
#include <cuda_runtime.h>
#include "tensor/device.h"

namespace tensor {

namespace {

inline void* CPUAllocSpace(size_t size, size_t alignment, Device device) {
  void *ptr;
#if _MSC_VER || defined(__MINGW32__) // MSVC, _WIN32 is for platform check
  ptr = _aligned_malloc(size, alignment);
#else
  int ret = posix_memalign(&ptr, alignment, nbytes);
  if (ret != 0) throw std::bad_alloc();
#endif
  return ptr;
}

inline void* CUDAAllocSpace(size_t size, size_t alignment, Device device) {
  void *dptr;
  CUDA_CALL(cudaSetDevice(device.id));
  CHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes\n";
  CUDA_CALL(cudaMalloc(&dptr, size));
  return dptr;
}

inline void CPUFreeSpace(void* dptr, Device device) {
#if _MSC_VER || defined(__MINGW32__)
  _aligned_free(dptr);
#else
  free(dptr);
#endif
}

inline void CUDAFreeSpace(void* dptr, Device device) {
  CUDA_CALL(cudaSetDevice(device.id));
  CUDA_CALL(cudaFree(dptr));
}

}

size_t Device::DeviceCount(DeviceType t) {
  int cnt = 0;
  cudaError_t e;

  switch (t) {
    case DeviceType::kCPU:
      return 1;
    case DeviceType::kCUDA:
      CUDA_CALL_WITH_ERROR_VAR(cudaGetDeviceCount(&cnt), e);
      return cnt;
    default: // kEmpty
      return 0;
  }
}

// return nullptr is allocation does not success
void* Device::AllocSpace(size_t size, size_t alignment, Device device) {
  CHECK(device.Valid()) << "Input device is not valid.\n";
  switch (device.type) {
    case DeviceType::kCPU:
      return CPUAllocSpace(size, alignment, device);
    case DeviceType::kCUDA:
      return CUDAAllocSpace(size, alignment, device);
    default:
      return nullptr;
  }
}

void Device::FreeSpace(void* dptr, Device device) {
  CHECK(device.Valid()) << "Input device is not valid.\n";
  if (!dptr) return;
  switch (device.type) {
    case DeviceType::kCPU:
      CPUFreeSpace(dptr, device);
      break;
    case DeviceType::kCUDA:
      CUDAFreeSpace(dptr, device);
      break;
    default:
      break;
  }
}

void Device::Transfer(const void* src,
                      Device src_device,
                      void* dst,
                      Device dst_device,
                      size_t size) {
  cudaError_t e;
  if (src_device.type == DeviceType::kCPU) {
    switch (dst_device.type) {
      case DeviceType::kCPU:
        std::memcpy(dst, src, size);
        break;
      case DeviceType::kCUDA:
        CUDA_CALL_WITH_ERROR_VAR(
          cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), e);
        break;
      default:
        break;
    }
  } else if (src_device.type == DeviceType::kCUDA) {
    switch (dst_device.type) {
      case DeviceType::kCPU:
        CUDA_CALL_WITH_ERROR_VAR(
          cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), e);
        break;
      case DeviceType::kCUDA:
      CUDA_CALL_WITH_ERROR_VAR(
          cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice), e);
        break;
      default:
        break;
    }
  }
}

} // namespace tensor


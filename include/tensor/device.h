/*!
 * \file device.h
 * \brief Device management and memory allocation tools
 */
#ifndef TENSOR_DEVICE_H_
#define TENSOR_DEVICE_H_

#include <string>
#include <cuda_runtime.h>
#include "utils/logging.h"
#include "tensor/macros.h"

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

#define CUDA_CALL_WITH_ERROR_VAR(func, e)                          \
  {                                                                \
    e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

namespace tensor {

enum class DeviceType : size_t {
  kCPU,
  kCUDA,
  kEmpty
};

inline std::string GetDeviceName(DeviceType deviceType) {
  switch (deviceType) {
    case DeviceType::kCPU:
      return "CPU";
    case DeviceType::kCUDA:
      return "CUDA";
    case DeviceType::kEmpty:
      return "Empty";
    default:
      return "UNKNOWN";
  }
}

struct Device final {
  Device() = default;
  Device(DeviceType i_type, int i_id) : type(i_type), id(i_id) {}

  bool operator==(const Device& other) const { return type == other.type && id == other.id; }
  bool operator!=(const Device& other) const { return !(*this == other); }
  bool Valid() const { return id < Device::DeviceCount(type); }
  bool IsEmpty() const { return type == DeviceType::kEmpty; }

  static Device EmptyDevice() { return Device(DeviceType::kEmpty, 0); }
  static Device DefaultDevice() { return Device(DeviceType::kCPU, 0); }

  TENSOR_DLL static size_t DeviceCount(DeviceType t);
  TENSOR_DLL static void* AllocSpace(size_t size, size_t alignment, Device device);
  TENSOR_DLL static void FreeSpace(void *dptr, Device device);
  TENSOR_DLL static void Transfer(
    const void* src, Device src_device, void* dst, Device dst_device, size_t size);

  static void SetCurrentDevice(Device device) {
    if (device.type != DeviceType::kCUDA || !device.Valid()) return;
    CUDA_CALL(cudaSetDevice(device.id));
  }

  static Device GetCurrentCUDADevice() {
    int ret = 0;
    CUDA_CALL(cudaGetDevice(&ret));
    return {DeviceType::kCUDA, ret};
  }

  DeviceType type;
  int id;
};

} // namespace tensor

#endif // TENSOR_DEVICE_H_

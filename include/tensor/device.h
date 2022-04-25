#ifndef TENSOR_DEVICE_H_
#define TENSOR_DEVICE_H_

#include <string>
#include "utils/logging.h"
#include "tensor/macros.h"

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

namespace tensor {

enum class DeviceType : size_t {
  kCPU,
  kCUDA
};

inline std::string GetDeviceName(DeviceType deviceType) {
  switch (deviceType) {
    case DeviceType::kCPU:
      return "CPU";
    case DeviceType::kCUDA:
      return "CUDA";
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

  TENSOR_DLL static size_t DeviceCount(DeviceType t);
  TENSOR_DLL static void* AllocSpace(size_t size, size_t alignment, Device device);
  TENSOR_DLL static void FreeSpace(void *dptr, Device device);

  DeviceType type;
  int id;
};

} // namespace tensor

#endif // TENSOR_DEVICE_H_

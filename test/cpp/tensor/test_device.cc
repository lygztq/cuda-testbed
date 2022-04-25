#include "tensor/device.h"
#include <gtest/gtest.h>

TEST(TestDevice, TestDeviceCase) {
  using tensor::Device;
  Device d(tensor::DeviceType::kCPU, 0);
  void* dptr = Device::AllocSpace(24, sizeof(int), d);
  Device::FreeSpace(dptr, d);
}

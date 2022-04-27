#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include "tensor/dtype.h"

static constexpr float kConvertErrFloat = static_cast<float>(1e-5);
static constexpr float kConvertErrFloatFiner = static_cast<float>(1e-10);

TEST(TestDType, TestFP16Size) {
  EXPECT_EQ(sizeof(fp16_t), 2);
}

TEST(TestDType, TestFP16To32) {
  // 0 01110 1111111111	
  fp16_t a_16 = fp16_t::FromHex(0x3bff);
  float a_32 = static_cast<float>(a_16);
  EXPECT_NEAR(a_32, 0.99951172f, kConvertErrFloatFiner);

  // 0 00000 0000000000
  fp16_t zero_16 = fp16_t::FromHex(0x0000);
  float zero_32 = static_cast<float>(zero_16);
  EXPECT_EQ(zero_32, 0.f);

  // 0 00000 0000000001
  fp16_t smallest_16 = fp16_t::FromHex(0x0001);
  float smallest_32 = static_cast<float>(smallest_16);
  EXPECT_NEAR(smallest_32, 0.000000059604645f, kConvertErrFloatFiner);

  // 0 00000 1111111111
  fp16_t largest_denorm_16 = fp16_t::FromHex(0x03ff);
  float largest_denorm_32 = static_cast<float>(largest_denorm_16);
  EXPECT_NEAR(largest_denorm_32, 0.000060975552f, kConvertErrFloatFiner);
  
  // 0 01111 0000000000 = 1
  fp16_t one_16 = fp16_t::FromHex(0x3c00);
  float one_32 = static_cast<float>(one_16);
  EXPECT_NEAR(one_32, 1.f, kConvertErrFloatFiner);

  // 0 11110 1111111111
  fp16_t max_16 = fp16_t::FromHex(0x7bff);
  float max_32 = static_cast<float>(max_16);
  EXPECT_NEAR(max_32, 65504.f, kConvertErrFloat);

  // 0 11111 0000000000 = inf
  fp16_t inf_16 = fp16_t::FromHex(0x7c00);
  float inf_32 = static_cast<float>(inf_16);
  EXPECT_EQ(inf_32, (std::numeric_limits<float>::infinity()));

  // 1 00000 0000000000 = -0
  fp16_t negzero_16 = fp16_t::FromHex(0x8000);
  float negzero_32 = static_cast<float>(negzero_16);
  EXPECT_EQ(negzero_32, -0.f);

  // 1 11111 0000000001 = nan
  fp16_t nan_16 = fp16_t::FromHex(0xfc01);
  float nan_32 = static_cast<float>(nan_16);
  EXPECT_TRUE(std::isnan(nan_32));
}

TEST(TestDType, TestFP32To16) {
  // 0
  float zero_32 = 0.f;
  fp16_t zero_16 = static_cast<fp16_t>(zero_32);
  EXPECT_EQ(static_cast<float>(zero_16), zero_32);

  // 1
  float one_32 = 1.f;
  fp16_t one_16 = static_cast<fp16_t>(one_32);
  EXPECT_EQ(static_cast<float>(one_16), one_32);

  // -0
  float negzero_32 = -0.f;
  fp16_t negzero_16 = static_cast<fp16_t>(negzero_32);
  EXPECT_EQ(static_cast<float>(negzero_16), negzero_32);

  // inf
  float inf_32 = std::numeric_limits<float>::infinity();
  fp16_t inf_16 = static_cast<fp16_t>(inf_32);
  EXPECT_EQ(std::numeric_limits<float>::infinity(), static_cast<float>(inf_16));

  // NaN
  float nan_32 = NAN;
  fp16_t nan_16 = static_cast<fp16_t>(nan_32);
  EXPECT_TRUE(std::isnan(static_cast<float>(nan_16)));

  // overflow
  float big_32 = 100000000.f;
  fp16_t big_16 = static_cast<fp16_t>(big_32);
  EXPECT_EQ(std::numeric_limits<float>::infinity(), static_cast<float>(big_16));

  // underflow
  float small_32 = static_cast<float>(1e-10);
  fp16_t small_16 = static_cast<fp16_t>(small_32);
  EXPECT_EQ(static_cast<float>(small_16), 0.f);
}

TEST(TestDType, TestTypeConvert) {
  // double to fp16
  fp16_t d_to_h = static_cast<fp16_t>(1.0);
  EXPECT_EQ(static_cast<double>(d_to_h), 1.0);

  // int to fp16
  fp16_t i_to_h = static_cast<fp16_t>(24);
  EXPECT_EQ(static_cast<int>(i_to_h), 24);
}

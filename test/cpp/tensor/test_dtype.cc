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
  fp16_t d_to_h = static_cast<fp16_t>((double)(1.0));
  EXPECT_EQ(static_cast<double>(d_to_h), 1.0);

  // int to fp16
  fp16_t i_to_h = static_cast<fp16_t>((int)(24));
  EXPECT_EQ(static_cast<int>(i_to_h), 24);
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4305)
#pragma warning(disable : 4244)
#pragma warning(disable : 4756)
#endif
TEST(TestDtype, TestScalar) {
  using tensor::Scalar;
  uint8_t uint8_val = 1;
  uint32_t uint32_val = 2;
  uint64_t uint64_val = 3;
  int8_t int8_val = 4;
  int32_t int32_val = 5;
  int64_t int64_val = 6;
  fp16_t fp16_val = 7.1;
  float float_val = 8.1f;
  double double_val = 9.1;
  uint64_t very_big_int_val = std::numeric_limits<uint64_t>::max();
  double very_big_float_val = std::numeric_limits<double>::max();

  Scalar uint8_scalar(uint8_val);
  Scalar uint32_scalar(uint32_val);
  Scalar uint64_scalar(uint64_val);
  Scalar int8_scalar(int8_val);
  Scalar int32_scalar(int32_val);
  Scalar int64_scalar(int64_val);
  Scalar fp16_scalar(fp16_val);
  Scalar float_scalar(float_val);
  Scalar double_scalar(double_val);
  Scalar very_big_int(very_big_int_val);
  Scalar very_big_float(very_big_float_val);
  
  // uint8 scalar
  EXPECT_EQ((int8_t)(uint8_scalar), (int8_t)(uint8_val));
  EXPECT_EQ((uint32_t)(uint8_scalar), (uint32_t)(uint8_val));
  EXPECT_EQ((uint64_t)(uint8_scalar), (uint64_t)(uint8_val));
  EXPECT_EQ((int32_t)(uint8_scalar), (int32_t)(uint8_val));
  EXPECT_EQ((int64_t)(uint8_scalar), (int64_t)(uint8_val));
  EXPECT_EQ((fp16_t)(uint8_scalar), (fp16_t)(uint8_val));
  EXPECT_EQ((float)(uint8_scalar), (float)(uint8_val));
  EXPECT_EQ((double)(uint8_scalar), (double)(uint8_val));
  EXPECT_EQ((uint8_t)(very_big_int), (uint8_t)(very_big_int_val));

  // fp16 scalar
  EXPECT_EQ((uint8_t)(fp16_scalar), (uint8_t)(fp16_val));
  EXPECT_EQ((int8_t)(fp16_scalar), (int8_t)(fp16_val));
  EXPECT_EQ((uint32_t)(fp16_scalar), (uint32_t)(fp16_val));
  EXPECT_EQ((uint64_t)(fp16_scalar), (uint64_t)(fp16_val));
  EXPECT_EQ((int32_t)(fp16_scalar), (int32_t)(fp16_val));
  EXPECT_EQ((int64_t)(fp16_scalar), (int64_t)(fp16_val));
  EXPECT_EQ((float)(fp16_scalar), (float)(fp16_val));
  EXPECT_EQ((double)(fp16_scalar), (double)(fp16_val));
  EXPECT_EQ((fp16_t)(very_big_float), (fp16_t)(very_big_float_val));

  // TODO: very small float underflow
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

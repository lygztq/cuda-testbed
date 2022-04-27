/*!
 * \file half.h
 */
#ifndef TENSOR_FP16_H_
#define TENSOR_FP16_H_

#include <fp16.h>

struct Half {
  uint16_t bits;

  Half() = default;
  Half(float f)
    : bits(fp16_ieee_from_fp32_value(f)) {}
  operator float() const { return fp16_ieee_to_fp32_value(bits); }

  static Half FromHex(uint16_t b) {
    Half h;
    h.bits = b;
    return h;
  }
};

using fp16_t = Half;

#endif  // TENSOR_FP16_H_

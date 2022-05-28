#include <limits>
#include <gtest/gtest.h>
#include "utils/random.h"

TEST(TestRandom, TestRandomGen) {
  using utils::RandomEngine;
  auto& engine = RandomEngine::ThreadLocal();

  engine.SetSeed(1);
  engine.RandInt32();
  engine.RandInt<uint64_t>(std::numeric_limits<uint64_t>::max());
  engine.Uniform<double>();
  engine.Normal<double>();
}

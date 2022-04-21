#include <gtest/gtest.h>
#include "tensor/function_ref.h"

namespace {

struct Add {
  int operator()(int a, int b) { return a + b; }
};

int add(int a, int b) { return a + b; }

Add makeAdd() { return Add(); }

} // namespace

TEST(TestFunctionRef, TestFunctionRefNormal) {
  tensor::function_ref<int(int, int)> add_ref(add);
  EXPECT_EQ(add_ref(4, 2), 6);
  
  // Directly use Add() here instead of makeAdd() will cause error, why?
  tensor::function_ref<int(int, int)> Add_ref(makeAdd());
  EXPECT_EQ(Add_ref(4, 2), 6);
}

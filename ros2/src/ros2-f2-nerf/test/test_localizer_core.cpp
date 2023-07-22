#include <gtest/gtest.h>
#include "../src/localizer_core.hpp"

TEST(LocalizerCoreTest, TestMonteCarloLocalize)
{
  LocalizerCoreParam param;
  param.runtime_config_path = "./runtime_config.yaml";
  LocalizerCore localizer_core(param);
  EXPECT_EQ(1, 1);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

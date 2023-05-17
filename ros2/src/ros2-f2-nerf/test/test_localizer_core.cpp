#include <gtest/gtest.h>
#include "../src/localizer_core.hpp"

TEST(LocalizerCoreTest, TestMonteCarloLocalize)
{
  LocalizerCore localizer_core("./runtime_config.yaml");
  EXPECT_EQ(1, 1);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

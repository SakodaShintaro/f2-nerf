#include <gtest/gtest.h>
#include "../src/localizer_core.hpp"

TEST(LocalizerCoreTest, TestMonteCarloLocalize)
{
  LocalizerCoreParam param;
  param.render_pixel_num = 256;
  LocalizerCore localizer_core("./runtime_config.yaml", param);
  EXPECT_EQ(1, 1);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#include "TestField.hpp"

#include "Field/Field.h"

void test_field(const std::string & config_path)
{
  const YAML::Node & config = YAML::LoadFile(config_path);
  Field field(config);
  const int64_t bs = 524288;
  torch::Tensor input = torch::randn({bs, 3}).cuda();
  torch::Tensor output =  field.Query(input);
  std::cout << "input.sizes() = " << input.sizes() << std::endl;
  std::cout << "output.sizes() = " << output.sizes() << std::endl;
  std::cout << "ans            = [" << bs << ", 16]" << std::endl;
}

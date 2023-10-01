#include "../field/hash_3d_anchored.hpp"
#include "../main_functions.hpp"

void test_field(const std::string & config_path)
{
  Hash3DAnchored field;
  const int64_t bs = 524288;
  torch::Tensor input = torch::randn({bs, 3}).cuda();
  torch::Tensor output = field.Query(input);
  std::cout << "input.sizes() = " << input.sizes() << std::endl;
  std::cout << "output.sizes() = " << output.sizes() << std::endl;
  std::cout << "ans            = [" << bs << ", 16]" << std::endl;
}

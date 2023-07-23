#include "src/ExpRunner.h"

#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, char * argv[])
{
  torch::manual_seed(2022);

  if (argc < 2) {
    std::cerr << "Please specify a config file path as command line argument." << std::endl;
    return 1;  // Return with error code
  }

  const std::string conf_path = argv[1];
  auto exp_runner = std::make_unique<ExpRunner>(conf_path);
  exp_runner->Execute();
}

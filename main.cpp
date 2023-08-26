#include "src/ExpRunner.h"
#include "src/main_functions.hpp"

#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, char * argv[])
{
  torch::manual_seed(2022);

  if (argc != 3) {
    std::cerr << "Please specify a config file path as command line argument." << std::endl;
    std::cerr << "argc = " << argc << std::endl;
    return 1;  // Return with error code
  }

  const std::string command = argv[1];
  const std::string conf_path = argv[2];
  if (command == "train") {
    auto exp_runner = std::make_unique<ExpRunner>(conf_path);
    exp_runner->Execute();
  } else if (command == "infer") {
    infer(conf_path);
  } else if (command == "walk") {
    walk(conf_path);
  } else if (command == "test") {
    test(conf_path);
  } else if (command == "test_field") {
    test_field(conf_path);
  } else {
    std::cerr << "Invalid command line argument : " << command << std::endl;
    return 1;
  }
}

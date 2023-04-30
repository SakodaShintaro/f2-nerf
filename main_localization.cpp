#include "src/localization_executer.hpp"

#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, char * argv[])
{
  std::cout << "Localization" << std::endl;
  std::string conf_path = "./runtime_config.yaml";
  std::unique_ptr<LocalizationExecuter> executer =
    std::make_unique<LocalizationExecuter>(conf_path);
  executer->Execute();
}

#include "src/main_functions/main_functions.hpp"
#include "src/main_functions/train_manager.hpp"

#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, char * argv[])
{
  torch::manual_seed(2022);

  if (argc < 2) {
    std::cerr << "Please specify a train_result_dir path as command line argument." << std::endl;
    std::cerr << "argc = " << argc << std::endl;
    return 1;
  }

  const std::string command = argv[1];
  const std::string train_result_dir = argv[2];
  const std::string dataset_dir = (argc >= 4 ? argv[3] : "");
  if (command == "train") {
    auto train_manager = std::make_unique<TrainManager>(train_result_dir, dataset_dir);
    train_manager->train();
  } else if (command == "infer") {
    infer(train_result_dir, dataset_dir);
  } else if (command == "walk") {
    walk(train_result_dir);
  } else if (command == "test") {
    test(train_result_dir, dataset_dir);
  } else {
    std::cerr << "Invalid command line argument : " << command << std::endl;
    return 1;
  }
}

#ifndef F2_NERF__MAIN_FUNCTIONS_HPP_
#define F2_NERF__MAIN_FUNCTIONS_HPP_

#include <string>

void walk(const std::string & train_result_dir);
void test(const std::string & train_result_dir, const std::string & dataset_dir);
void infer(const std::string & train_result_dir, const std::string & dataset_dir);

#endif  // F2_NERF__MAIN_FUNCTIONS_HPP_

//
// Created by ppwang on 2022/5/18.
//
#pragma once
#include <chrono>
#include <string>

class StopWatch {
public:
  StopWatch();
  ~StopWatch() = default;
  double TimeDuration();
  std::chrono::steady_clock::time_point t_point_;
};

class ScopeWatch {
public:
  ScopeWatch(const std::string& scope_name);
  ~ScopeWatch();
  std::chrono::steady_clock::time_point t_point_;
  std::string scope_name_;
};

class Timer
{
public:
  void start() { start_time_ = std::chrono::steady_clock::now(); }
  int64_t elapsed_milli_seconds() const
  {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  }
  double elapsed_seconds() const { return elapsed_milli_seconds() / 1000.0; }

private:
  std::chrono::steady_clock::time_point start_time_;
};

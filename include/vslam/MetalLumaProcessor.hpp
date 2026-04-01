#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace vslam {

class MetalLumaProcessor {
 public:
  MetalLumaProcessor();
  ~MetalLumaProcessor();

  MetalLumaProcessor(MetalLumaProcessor&&) noexcept;
  MetalLumaProcessor& operator=(MetalLumaProcessor&&) noexcept;

  MetalLumaProcessor(const MetalLumaProcessor&) = delete;
  MetalLumaProcessor& operator=(const MetalLumaProcessor&) = delete;

  bool isAvailable() const;
  const std::string& lastError() const;
  bool convertBgrToGray(const cv::Mat& bgr, cv::Mat& gray);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace vslam

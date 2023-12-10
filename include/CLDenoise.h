/*
Copyright [2023] <Copyright LZY>
*/
#ifndef INCLUDE_CLDENOISE_H_
#define INCLUDE_CLDENOISE_H_
#include <utility>
#include <memory>
#include <string>
// torch header
#include "torch/script.h"
#include "torch/torch.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
namespace neo {
cv::Mat addNoise(cv::Mat Image);
// 图像去噪的操作类
class DenoiseOP {
 private:
  cv::Mat m_image_;
  std::pair<int, int> m_image_size_;
  std::string m_model_path_ = {"../../models/libtorch-model-gpu.pt"};
  torch::DeviceType m_device_type_ = {torch::kCPU};
  std::shared_ptr<torch::jit::script::Module> m_model_;
  bool m_is_denoised_ = false;

 public:
  DenoiseOP();
  explicit DenoiseOP(const std::string& model_path);
  ~DenoiseOP();
  void LoadImage(const std::string& image_path);
  void LoadImage(cv::Mat image);
  void Denoise();
  void DenoiseUML();
  cv::Mat GetImage();
};
}  // namespace neo
#endif  // INCLUDE_CLDENOISE_H_

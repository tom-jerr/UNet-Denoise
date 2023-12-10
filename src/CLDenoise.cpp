/*
Copyright [2023] <Copyright LZY>
*/
#include <iostream>
#include <random>
#include <utility>
// torch heade
#include "torch/script.h"
#include "torch/torch.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../include/CLDenoise.h"
namespace neo {
cv::Mat addNoise(cv::Mat Image) {
  // cv::Mat Image = image.clone();
  // 添加随机噪声
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 25);  // 均值为0，标准差为25的正态分布

  for (int y = 0; y < Image.rows; y++) {
    for (int x = 0; x < Image.cols; x++) {
      for (int c = 0; c < Image.channels();
           c++) {  // 对于RGBA图像，需要遍历4个通道
        double noise = d(gen);
        uchar &pixel = Image.at<cv::Vec4b>(y, x)[c];
        pixel = cv::saturate_cast<uchar>(pixel + noise);
      }
    }
  }
  return Image;
}

DenoiseOP::DenoiseOP() {
  m_model_ = std::make_shared<torch::jit::script::Module>(m_model_path_);
}

DenoiseOP::DenoiseOP(const std::string& model_path) {
  std::cout << "load model is beginning\n";
  m_model_path_ = model_path;
  m_model_ = std::make_shared<torch::jit::script::Module>(m_model_path_);
  std::cout << "model is loaded\n";
}

DenoiseOP::~DenoiseOP() {
  if (m_image_.data) m_image_.release();
}

void DenoiseOP::LoadImage(const std::string& image_path) {
  m_image_ = cv::imread(image_path, cv::IMREAD_COLOR);
  int width = m_image_.rows;
  int height = m_image_.cols;
  m_image_size_ = std::make_pair(width, height);
  int n = width > height ? height : width;
  cv::resize(m_image_, m_image_, cv::Size(n, n));
}

void DenoiseOP::LoadImage(cv::Mat image) {
  m_image_ = std::move(image);
  int width = m_image_.rows;
  int height = m_image_.cols;
  m_image_size_ = std::make_pair(width, height);
  int n = width > height ? height : width;
  cv::resize(m_image_, m_image_, cv::Size(n, n));
}

void DenoiseOP::Denoise() {
  auto m_device = torch::Device(m_device_type_);
  if (!m_image_.data) {
    std::cout << "Error: image is empty!" << std::endl;
    return;
  }
  cv::Mat im = m_image_;
  // convert to float and normalize
  im.convertTo(im, CV_32FC3, 1.F / 255.F);
  int img_width = im.cols;
  int img_height = im.rows;
  int img_channels = im.channels();
  torch::TensorOptions options(torch::kFloat32);
  // transform cv::Mat to tensor
  auto img_tensor = torch::from_blob(
      im.data, {1, img_height, img_width, img_channels}, options);
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  auto img_var = torch::autograd::make_variable(img_tensor, false).to(m_device);
  // input tensor
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(img_var);
  cv::Mat img_post;
  at::Tensor output;
  try {
    output = m_model_->forward(inputs).toTensor().to(torch::kCPU).squeeze();
  } catch (const c10::Error &e) {
    std::cout << "Error: " << e.msg() << std::endl;
    return;
  }
  output = output.permute({1, 2, 0}).clamp(0, 1).mul(255).to(torch::kU8);
  std::cout << output.sizes() << std::endl;
  img_post = cv::Mat(img_height, img_width, CV_8UC3, output.data_ptr());

  m_image_ = img_post;
  m_is_denoised_ = true;

  cv::imwrite("../pic/result.jpg", img_post);
  cv::imshow("enhanced img", img_post);
}

void DenoiseOP::DenoiseUML() {
  cv::Mat kernal = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::Mat result;
  cv::Mat dst;

  if (!m_image_.data) {
    std::cout << "Error: image is empty!" << std::endl;
    return;
  }
  if (m_is_denoised_) {
    std::cout << "Image has been denoised! Just UML" << std::endl;
    // USM sharpening

    cv::bilateralFilter(m_image_, dst, 5, 120, 3);

    cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);
    cv::imwrite("../../pic/result.png", result);
    cv::imshow("dst", result);
    cv::waitKey(0);
    return;
  }
  Denoise();
  // USM sharpening

  cv::bilateralFilter(m_image_, dst, 5, 120, 3);

  cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);
  cv::imwrite("../../pic/result.png", result);
  cv::imshow("dst", result);
  // cv::waitKey(0);
}

cv::Mat DenoiseOP::GetImage() {
  if (!m_image_.data) {
    throw std::runtime_error("Error: image is empty!");
  }
  if (m_image_size_.first != 0 && m_image_size_.second != 0) {
    cv::resize(m_image_, m_image_,
               cv::Size(m_image_size_.first, m_image_size_.second));
  }
  return m_image_;
}

}  // namespace neo

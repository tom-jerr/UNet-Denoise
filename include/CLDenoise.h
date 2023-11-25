#ifndef CLDENOISE_H_
#define CLDENOISE_H_
// torch header
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <memory>
#include <thread>

// 线程数
#define THREAD_NUM 10
// 模型路径和模型
std::string g_model_path = "../../models/libtorch-model-gpu.pt";
std::shared_ptr<torch::jit::script::Module> g_model;

// 接收前端传来的图片
std::vector<cv::Mat> g_img_vec;
std::vector<std::thread> g_thread_vec;


// 图像去噪的操作类
class DenoiseOP {
  private:
    cv::Mat m_image_;
    std::string m_model_path_ = {"../../models/libtorch-model-gpu.pt"};
    torch::DeviceType m_device_type_ = {torch::kCPU};
    //std::shared_ptr<torch::jit::script::Module> m_model_;
    bool m_is_denoised_ = false;
  public:

    DenoiseOP();
    //DenoiseOP(const std::string model_path);
    ~DenoiseOP();
    void LoadImage(std::string image_path);
    void LoadImage(cv::Mat image);
    void Denoise();
    void DenoiseUML();
    cv::Mat GetImage();
};

DenoiseOP::DenoiseOP(){
    /*m_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(m_model_path_));*/
}

//DenoiseOP::DenoiseOP(const std::string model_path) {
//  std::cout << "load model is beginning\n";
//  m_model_path_ = model_path;
//  m_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(m_model_path_));
//  std::cout << "model is loaded\n";
//
//}

DenoiseOP::~DenoiseOP() {
  if (m_image_.data) 
    m_image_.release();
}

void DenoiseOP::LoadImage(std::string image_path) {
  m_image_ = cv::imread(image_path, cv::IMREAD_COLOR);
}

void DenoiseOP::LoadImage(cv::Mat image) {
  m_image_ = image;
}

void DenoiseOP::Denoise() {
  torch::Device m_device = torch::Device(m_device_type_);
  if (!m_image_.data) {
    std::cout << "Error: image is empty!" << std::endl;
    return;
  }
  cv::Mat im = m_image_;
  // convert to float and normalize
  im.convertTo(im, CV_32FC3, 1.f / 255.f);
  int img_width = im.cols;
  int img_height = im.rows;
  int img_channels = im.channels();
  torch::TensorOptions options(torch::kFloat32);
  // transform cv::Mat to tensor
  auto img_tensor = torch::from_blob(im.data, { 1, img_height, img_width, img_channels }, options);
  img_tensor = img_tensor.permute({ 0,3,1,2 });
  auto img_var = torch::autograd::make_variable(img_tensor, false).to(m_device);
  // input tensor
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(img_var);
  cv::Mat img_post;
  at::Tensor output;
  try {
    output = g_model->forward(inputs).toTensor().to(torch::kCPU).squeeze();
  } catch (const c10::Error& e) {
    std::cout << "Error: " << e.msg() << std::endl;
    return;
  }
  output = output.permute({1,2,0}).clamp(0,1).mul(255).to(torch::kU8);
  std::cout << output.sizes() << std::endl;
  img_post = cv::Mat(img_height, img_width, CV_8UC3, output.data_ptr());
  
 
  m_image_ = img_post;
  m_is_denoised_ = true;

  std::string suffix = ".jpg";
  std::string save_path = "../pic/result"+ + std::rand() + suffix;
  cv::imwrite("../pic/result.jpg", img_post);
  cv::imshow("enhanced img", img_post);
  // cv::waitKey(0);
}

void DenoiseOP::DenoiseUML() {
  cv::Mat kernal = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cv::Mat result, dst;

  if (!m_image_.data) {
    std::cout << "Error: image is empty!" << std::endl;
    return;
  }
  if(m_is_denoised_) {
    std::cout << "Image has been denoised! Just UML" << std::endl;
    // USM sharpening
    
    cv::bilateralFilter(m_image_, dst, 5, 120, 3);

    cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);
    cv::imwrite("../../pic/result.png", result);
    //cv::imshow("dst", result);
    //cv::waitKey(0);
    return;
  }
  Denoise();
  // USM sharpening

  cv::bilateralFilter(m_image_, dst, 5, 120, 3);

  cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);
  cv::imwrite("../../pic/denoise_result.png", result);
  std::cout << "denoise result is saved" << std::endl;      
  //cv::imshow("dst", result);
  // cv::waitKey(0);
}

cv::Mat DenoiseOP::GetImage() {
  if (!m_image_.data) {
	std::runtime_error("Error: image is empty!");
  }
  return m_image_;
}
void MainLoopDenoise();
void InitModel();

#endif /* CLDENOISE_H_ */
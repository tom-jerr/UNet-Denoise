#pragma once

#ifndef CLDENOISE_H_
#define CLDENOISE_H_
// torch header
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// #include <vector>
// #include <string>
// #include <memory>
// #include <thread>


// // 模型路径和模型
// std::string g_model_path = "model/libtorch-model-gpu.pt";
// std::shared_ptr<torch::jit::script::Module> g_model;

// // 接收前端传来的图片
// std::vector<cv::Mat> g_img_vec;
// std::vector<std::thread> g_thread_vec;


// 图像去噪的操作类
class DenoiseOP {
  private:
    cv::Mat m_image_;
    std::string m_model_path_ = {"model/libtorch-model-gpu.pt"};
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

extern cv::Mat m_image_return;
void Init();
void MainLoopDenoise(std::string fileName);
void InitModel();
void logOP();

#endif /* CLDENOISE_H_ */

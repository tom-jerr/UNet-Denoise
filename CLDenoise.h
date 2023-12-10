#pragma once

#ifndef INCLUDE_CLDENOISE_H_ 
#define INCLUDE_CLDENOISE_H_ 
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
    cv::Mat m_image_;                                 // 存放需要去噪的图片
    torch::DeviceType m_device_type_ = {torch::kCPU}; // 选择设备类型
    bool m_is_denoised_ = false;                      // 是否已经去噪
    // std::string m_model_path_ = {"model/libtorch-model-gpu.pt"};
    //std::shared_ptr<torch::jit::script::Module> m_model_;
  public:

    DenoiseOP();                                    
    //DenoiseOP(const std::string model_path);
    ~DenoiseOP();
    void LoadImage(std::string image_path);         // 加载图片
    void LoadImage(cv::Mat image);                  // 加载图片
    void Denoise();                                 // Unet模型去噪
    void DenoiseUML();                              // Unet模型+UML锐化去噪
    cv::Mat GetImage();                             // 获取去噪后的图片
};

extern cv::Mat m_image_return;
void Init();                                      // 初始化模型以及日志
void MainLoopDenoise(std::string fileName);       // 去噪主循环
void logOP();
// void InitModel(); 
#endif  // INCLUDE_CLDENOISE_H_ 

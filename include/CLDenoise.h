#ifndef CLDENOISE_H_
#define CLDENOISE_H_
// torch header
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
namespace neo{
cv::Mat add_noise(cv::Mat Image);
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
    DenoiseOP(const std::string model_path);
    ~DenoiseOP();
    void LoadImage(std::string image_path);
    void LoadImage(cv::Mat image);
    void Denoise();
    void DenoiseUML();
    cv::Mat GetImage();
};

} // namespace neo


#endif /* CLDENOISE_H_ */
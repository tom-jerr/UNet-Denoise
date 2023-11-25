#include <iostream>

// torch heade
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/CLDenoise.h"


extern std::string g_model_path;
extern std::shared_ptr<torch::jit::script::Module> g_model;
extern std::vector<cv::Mat> g_img_vec;
extern std::vector<std::thread> g_thread_vec;
std::string img_path = "../../pic/noise_Testing.png";
// 加载模型
void InitModel() {
	g_model = std::make_shared<torch::jit::script::Module>(torch::jit::load(g_model_path));
}

void MainLoopDenoise() {
    
    if (g_model == nullptr) {
        std::cout << "model is nullptr" << std::endl;
        std::runtime_error("model is nullptr\n");
    }
    int flag = 0;
    for (;;) {
        // 收到前端获得的图片

        // 向g_img_vec中添加图片
        if (flag == 0) {
            g_img_vec.push_back(cv::imread(img_path, cv::IMREAD_COLOR));
            flag++;
        }
        // 如果g_img_vec中有图片，新建线程进行图片去噪
        while (g_img_vec.size() > 0) {
            std::cout << "g_img_vec size: " << g_img_vec.size() << std::endl;
			DenoiseOP denoise_op;
            denoise_op.LoadImage(g_img_vec[0]);
            std::thread denoise_thread(&DenoiseOP::DenoiseUML, &denoise_op);
            denoise_thread.join();

			g_img_vec.erase(g_img_vec.begin());
		}
        
    }
}

int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    std::cout << "model path: " << g_model_path << std::endl;
    std::cout << "img path: " << img_path << std::endl;

    std::cout << "init model" << std::endl;
    InitModel();

    std::cout << "main loop begin" << std::endl;
    MainLoopDenoise();
    return 0;
}


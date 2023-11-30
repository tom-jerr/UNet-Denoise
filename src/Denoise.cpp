#include <iostream>
#include <stdio.h>
// torch heade
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/CLDenoise.h"
#include "../include/CLLog.h"

// 全局变量
extern std::string g_model_path;
extern std::shared_ptr<torch::jit::script::Module> g_model;
extern std::vector<cv::Mat> g_img_vec;
extern std::vector<std::thread> g_thread_vec;
std::string img_path = "../../pic/noise_Testing.png";
std::string log_path = "../../Log/DenoiseLog.txt";  
//FILE* fp = fopen(log_path.c_str(), "a");
Log* g_log;
// 去噪初始化
void Init() {
	g_model = std::make_shared<torch::jit::script::Module>(torch::jit::load(g_model_path));
    g_log = Log::getLogInstance(log_path);   
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
            time_t begin = time(0);
            std::thread denoise_thread(&DenoiseOP::DenoiseUML, &denoise_op);
            denoise_thread.join();
            // 记录日志
            time_t end = time(0);
            std::cout << "Denoise time: " << end - begin << "s" << std::endl;
            std::string log_msg = "Denoise time: " + std::to_string(end - begin) + "s";
            g_log->writeLog(log_msg.c_str());
            // 将去噪后的图片发送给前端
			g_img_vec.erase(g_img_vec.begin());
		}
        
    }
}

int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    std::cout << "model path: " << g_model_path << std::endl;
    std::cout << "img path: " << img_path << std::endl;
    //fprintf(fp, "global model path: %s\n", g_model_path.c_str());
    //fprintf(fp, "global img path: %s\n", img_path.c_str());
    std::cout << "init model" << std::endl;
    Init();
    g_log->writeLog("Denoise success");
    std::cout << "main loop begin" << std::endl;
    MainLoopDenoise();
    return 0;
}


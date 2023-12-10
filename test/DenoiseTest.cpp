#include <iostream>
#include <time.h>
// torch heade
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/CLDenoise.h"
//#include "../include/ThreadPool.h"
using namespace torch;
using namespace std;
using namespace cv;

extern std::string g_model_path;
extern std::shared_ptr<torch::jit::script::Module> g_model;
extern std::vector<cv::Mat> g_img_vec;
extern std::vector<std::thread> g_thread_vec;

// ����ģ��
void InitModel() {
	g_model = std::make_shared<torch::jit::script::Module>(torch::jit::load(g_model_path));
}

int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    string model_path = "../../models/libtorch-model-gpu.pt";
    string img_path = "../../pic/noise_Testing.png";
    
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    cout << "model path: " << model_path << endl;
    cout << "img path: " << img_path << endl;

    cout << "init model" << endl;
    InitModel();

    cout << "main loop begin" << endl;
    if (g_model == nullptr) {
        cout << "model is nullptr" << endl;
        std::runtime_error("model is nullptr\n");
    }
    int flag;
    for (;;) {
        // �յ�ǰ�˻�õ�ͼƬ

        // ��g_img_vec������ͼƬ
        if (flag == 0) {
            g_img_vec.push_back(cv::imread(img_path, cv::IMREAD_COLOR));
            flag++;
        }
        // ���g_img_vec����ͼƬ���½��߳̽���ͼƬȥ��
        while (g_img_vec.size() > 0) {
			cout << "g_img_vec size: " << g_img_vec.size() << endl;
			DenoiseOP denoise_op;
            time_t start = time(0);
            denoise_op.LoadImage(g_img_vec[0]);
            std::thread denoise_thread(&DenoiseOP::DenoiseUML, &denoise_op);
            denoise_thread.join();
            time_t end = time(0);
            std::cout << "Denoise time: " << end - start << "s" << std::endl;
			g_img_vec.erase(g_img_vec.begin());
		}
        
    }
  
    return 0;
}


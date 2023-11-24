#include <iostream>

// torch heade
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/CLDenoise.h"
using namespace torch;
using namespace std;
using namespace cv;
//std::string g_model_path = "../../models/libtorch-model-gpu.pt";
//std::shared_ptr<torch::jit::script::Module> g_model = std::make_shared<torch::jit::script::Module>(g_model_path);
int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    string model_path = "./libtorch-model-gpu.pt";
    string img_path = "../../pic/noise_Testing.png";
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    cout << "model path: " << model_path << endl;
    cout << "img path: " << img_path << endl;
    DenoiseOP denoise_op(model_path);

    cout << "load image" << endl;
    denoise_op.LoadImage(img_path);
    cv::Mat img_post = denoise_op.GetImage();
    imshow("src img", img_post);

    cout << "denoise" << endl;
    denoise_op.DenoiseUML();   
  
    waitKey(0);

    return 0;
}


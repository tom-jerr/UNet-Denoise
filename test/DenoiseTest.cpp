#include <iostream>
#include <unistd.h>
// torch header
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
using namespace neo;

int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    string model_path = "libtorch-model-gpu.pt";
    string img_path = "img.png";
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    cout << "model path: " << model_path << endl;
    cout << "img path: " << img_path << endl;

    cv::Mat img = cv::imread(img_path);
    cv::Mat noise_img = add_noise(img);

    DenoiseOP denoise_op(model_path);
    cout << "load image" << endl;
    denoise_op.LoadImage(noise_img);
    // cv::Mat img_post = denoise_op.GetImage();
    imshow("src img", noise_img);

    cout << "denoise" << endl;
    denoise_op.DenoiseUML();   
  
    waitKey(0);

    return 0;
}


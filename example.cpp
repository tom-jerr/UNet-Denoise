#include <iostream>

// torch相关引用
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV相关引用
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace torch;
using namespace std;
using namespace cv;

int main() {
    
    string model_path = "C:/Users/liuzhiyi/Desktop/libtorch_test/demo/libtorch-model-gpu.pt";
    string img_path = "C:/Users/liuzhiyi/Desktop/libtorch_test/demo/noise_Testing.png";

    // 加载模型
    cout << "Loading low-light model" << endl;
    std::shared_ptr<torch::jit::script::Module> module = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
    cout << "Initialized low-light model" << endl;

    torch::DeviceType device_type;
    device_type = torch::kCPU;
    torch::Device device(device_type);
    cout << "reading image" << endl;
    Mat im = imread(img_path, IMREAD_COLOR);
    imshow("src img", im);
    //cvtColor(im, im, COLOR_BGR2RGB);
    //Mat normedImg;
    //// 影像灰度归一化，img是OpenCV的float32的Mat类型
    im.convertTo(im, CV_32FC3, 1.f / 255.f);

 /*   imshow("src BGR img", im);*/
    int img_width = im.cols;
    int img_height = im.rows;
    int img_channels = im.channels();
    torch::TensorOptions options(torch::kFloat32);
    // 将OpenCV的Mat类型构造成Tensor，然后再转换成可求导的变量
    auto img_tensor = torch::from_blob(im.data, { 1, img_height, img_width, img_channels }, options);
    img_tensor = img_tensor.permute({ 0,3,1,2 });
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

 
    //// 前向推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);


    cv::Mat img_post;
    at::Tensor output;
    try {
        cout << "unet predict beginning\n";
        output = module->forward(inputs).toTensor().to(torch::kCPU).squeeze();
        cout << "unet predict done\n";  
        cout << output.sizes() << endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    

    // 后处理
    output = output.permute({1,2,0}).clamp(0,1).mul(255).to(torch::kU8);
    try {
        cv::Mat final_img(img_width, img_height, CV_8UC3, output.data_ptr());
        imshow("enhanced img", final_img);
       /* cv::Mat img;
        cvtColor(final_img, img, COLOR_GRAY2BGR);
        imshow("res", img);*/
    }
    catch (exception e)
    {
        cout << e.what() << endl;
    }
    
  
    waitKey(0);

    return 0;
}


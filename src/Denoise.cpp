#include <iostream>

// torch heade
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace torch;
using namespace std;
using namespace cv;

int main() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    string model_path = "../../models/libtorch-model-gpu.pt";
    string img_path = "../../pic/noise_Testing.png";

    // load pytorch model
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
    //// convert to float and normalize
    im.convertTo(im, CV_32FC3, 1.f / 255.f);

 /*   imshow("src BGR img", im);*/
    int img_width = im.cols;
    int img_height = im.rows;
    int img_channels = im.channels();
    torch::TensorOptions options(torch::kFloat32);
    // transform cv::Mat to tensor
    auto img_tensor = torch::from_blob(im.data, { 1, img_height, img_width, img_channels }, options);
    img_tensor = img_tensor.permute({ 0,3,1,2 });
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

 
    
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
    

    // transfer float tensor to uint tensor and prepare to cv::Mat
    output = output.permute({1,2,0}).clamp(0,1).mul(255).to(torch::kU8);
    try {
        cv::Mat final_img(img_width, img_height, CV_8UC3, output.data_ptr());

        imshow("enhanced img", final_img);

        // USM sharpening
        Mat kernal = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        Mat result, dst;

        bilateralFilter(final_img, dst, 5, 120, 3);

        filter2D(dst, result, -1, kernal, Point(-1, -1), 0);
        imwrite("../../pic/result.png", result);
        imshow("dst", result);
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


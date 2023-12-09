#include <iostream>

#include "CLDenoise.h"
#include "CLLog.h"
//#include "../include/CLThreadPool.h"
// 线程数
#define THREAD_NUM 2
// 全局变量
// extern std::string g_model_path;
// extern std::shared_ptr<torch::jit::script::Module> g_model;
// extern std::vector<cv::Mat> g_img_vec;
// extern std::vector<std::thread> g_thread_vec;

// 模型路径和模型
std::string g_model_path = "./model/libtorch-model-gpu.pt";
std::shared_ptr<torch::jit::script::Module> g_model;

// 接收前端传来的图片
std::vector<cv::Mat> g_img_vec;
std::vector<std::thread> g_thread_vec;

std::string img_path = "./pic/noise_Testing.png";
std::string log_path = "./Log/DenoiseLog.txt";
//FILE* fp = fopen(log_path.c_str(), "a");
Log* g_log;
//ThreadPool g_pool(THREAD_NUM);

cv::Mat m_image_return;

// 去噪初始化
void Init() {
    g_model = std::make_shared<torch::jit::script::Module>(torch::jit::load(g_model_path));
    g_log = Log::getLogInstance(log_path);
}

void MainLoopDenoise(std::string fileName) {
    
    if (g_model == nullptr) {
        std::cout << "model is nullptr" << std::endl;
        std::runtime_error("model is nullptr\n");
    }
    int flag = 0;
    for (;;) {
        // 收到前端获得的图片
        img_path = fileName;

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
        return;
    }
}

//void ThreadPoolDenoise() {
//    DenoiseOP* op = new DenoiseOP();
//    op->LoadImage(img_path);
//    g_pool.AddTask(op);
//}

void logOP() {
    // C:/Users/liuzhiyi/Desktop/UnetDenoise/
    
    //std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(model_path);
    std::cout << "model path: " << g_model_path << std::endl;
    std::cout << "img path: " << img_path << std::endl;
    //fprintf(fp, "global model path: %s\n", g_model_path.c_str());
    //fprintf(fp, "global img path: %s\n", img_path.c_str());
    std::cout << "init model" << std::endl;
    Init();
    g_log->writeLog("Denoise success");
    std::cout << g_log->getLogFileName()<< std::endl;
    std::cout << "main loop begin" << std::endl;
    /*MainLoopDenoise();*/
    //ThreadPoolDenoise();
    //return 0;
}

DenoiseOP::DenoiseOP(){
    /*m_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(m_model_path_));*/
}

//DenoiseOP::DenoiseOP(const std::string model_path) {
//  std::cout << "load model is beginning\n";
//  m_model_path_ = model_path;
//  m_model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(m_model_path_));
//  std::cout << "model is loaded\n";
//
//}

DenoiseOP::~DenoiseOP() {
    if (m_image_.data)
        m_image_.release();
}

void DenoiseOP::LoadImage(std::string image_path) {
    m_image_ = cv::imread(image_path, cv::IMREAD_COLOR);
}

void DenoiseOP::LoadImage(cv::Mat image) {
    m_image_ = image;
}

void DenoiseOP::Denoise() {
    torch::Device m_device = torch::Device(m_device_type_);
    if (!m_image_.data) {
        std::cout << "Error: image is empty!" << std::endl;
        return;
    }
    cv::Mat im = m_image_;
    // convert to float and normalize
    im.convertTo(im, CV_32FC3, 1.f / 255.f);
    int img_width = im.cols;
    int img_height = im.rows;
    int img_channels = im.channels();
    torch::TensorOptions options(torch::kFloat32);
    // transform cv::Mat to tensor
    auto img_tensor = torch::from_blob(im.data, { 1, img_height, img_width, img_channels }, options);
    img_tensor = img_tensor.permute({ 0,3,1,2 });
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(m_device);
    // input tensor
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);
    cv::Mat img_post;
    at::Tensor output;
    try {
        output = g_model->forward(inputs).toTensor().to(torch::kCPU).squeeze();
    } catch (const c10::Error& e) {
        std::cout << "Error: " << e.msg() << std::endl;
        return;
    }
    output = output.permute({1,2,0}).clamp(0,1).mul(255).to(torch::kU8);
    std::cout << output.sizes() << std::endl;
    img_post = cv::Mat(img_height, img_width, CV_8UC3, output.data_ptr());


    m_image_ = img_post;
    m_is_denoised_ = true;

    std::string suffix = ".jpg";
    std::string save_path = "./pic/result" + std::rand() + suffix;
    cv::imwrite("pic/result.jpg", img_post);
    cv::imshow("enhanced img", img_post);
    // cv::waitKey(0);
}

void DenoiseOP::DenoiseUML() {
    cv::Mat kernal = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::Mat result, dst;

    if (!m_image_.data) {
        std::cout << "Error: image is empty!" << std::endl;
        return;
    }
    if(m_is_denoised_) {
        std::cout << "Image has been denoised! Just UML" << std::endl;
        // USM sharpening

        cv::bilateralFilter(m_image_, dst, 5, 120, 3);

        cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);
        cv::imwrite("./pic/result.png", result);
        //cv::imshow("dst", result);
        //cv::waitKey(0);
        return;
    }
    Denoise();
    // USM sharpening

    cv::bilateralFilter(m_image_, dst, 5, 120, 3);

    cv::filter2D(dst, result, -1, kernal, cv::Point(-1, -1), 0);

    cv::imwrite("./pic/denoise_result.png", result);
    std::cout << "denoise result is saved" << std::endl;
    //cv::imshow("dst", result);
    // cv::waitKey(0);
}

cv::Mat DenoiseOP::GetImage() {
    if (!m_image_.data) {
        std::runtime_error("Error: image is empty!");
    }
    return m_image_;
}



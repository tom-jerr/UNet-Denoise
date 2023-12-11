/*
Copyright [2023] <Copyright LZY>
*/
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>

// torch header
#include "torch/script.h"
#include "torch/torch.h"

// OpenCV header
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "../include/CLDenoise.h"
// using namespace torch;
// using namespace std;
// using namespace cv;
// using namespace neo;

int main() {
  // C:/Users/liuzhiyi/Desktop/UnetDenoise/
  std::string model_path = "libtorch-model-gpu.pt";
  std::string img_path = "img.png";
  std::string log_path = "../../log/c++log.txt";
  // std::shared_ptr<torch::jit::script::Module> model =
  // std::make_shared<torch::jit::script::Module>(model_path);
  std::cout << "model path: " << model_path << std::endl;
  std::cout << "img path: " << img_path << std::endl;
  FILE *fp = fopen(log_path.c_str(), "w");
  cv::Mat img = cv::imread(img_path);
  cv::Mat noise_img = neo::addNoise(img);

  neo::DenoiseOP denoise_op(model_path);
  std::cout << "load image" << std::endl;

  // cv::Mat img_post = denoise_op.GetImage();
  // imshow("src img", noise_img);

  struct timeval start;
  struct timeval end;
  gettimeofday(&start, nullptr);
  std::cout << "denoise" << std::endl;
  // for (int i = 0; i < 50; i++){
  //   denoise_op.LoadImage(noise_img);
  //   denoise_op.DenoiseUML();
  // }
  denoise_op.LoadImage(noise_img);
  denoise_op.DenoiseUML();
  gettimeofday(&end, nullptr);
  std::cout << (end.tv_sec - start.tv_sec) * 1000 +
                   (end.tv_usec - start.tv_usec) / 1000
            << "ms" << std::endl;
  
  fprintf(fp, "time: %ldms\n", (end.tv_sec - start.tv_sec) * 1000 +
                                   (end.tv_usec - start.tv_usec) / 1000);
  cv::waitKey(0);
  fclose(fp);
  return 0;
}

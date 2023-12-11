# 图像去噪系统
- 基于windows 10/11
- 后端算法(master分支)：基于Unet的去噪模型+UML锐化算法
- model-ubuntu：是在WSL ubuntu20.04上面进行模型测试和训练的分支
  - 模型可以通过pytorch，libtorch，CUDA等第三方库实现跨平台的应用

## 前端
- 基于QT进行页面构建

## 后端
- 将pytorch训练的模型，使用libtorch加载到C++中

## 问题
- 基于QT的pro项目可以正常运行；但是基于QT的CMake会提示找不到nvcc编译器
  - 尝试多种解决方案失败后，采用pro的项目结构
- 现有的处理图片方式，是等待线程执行完才会进行下一轮去噪；考虑加入线程池，将去噪和锐化分开进行，增加并行化
  
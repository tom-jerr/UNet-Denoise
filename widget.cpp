#include "Widget.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPixmap>
#include <random>
#include "CLDenoise.h"
//#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
    this->resize(1366,768);
    this->setWindowTitle("图像去噪");

    // 创建主布局
    QHBoxLayout *mainLayout = new QHBoxLayout(this);

    // 创建左侧布局，用于放置原图和打开图片按钮
    QVBoxLayout *leftLayout = new QVBoxLayout();
    QPushButton *button = new QPushButton("打开图片", this);
    connect(button, &QPushButton::clicked, this, &MainWindow::onButtonClicked);
    originalImageLabel = new QLabel(this);
    originalImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    leftLayout->addWidget(button);
    leftLayout->addWidget(originalImageLabel);

    // 创建增加噪声按钮
    QVBoxLayout *middleLayout = new QVBoxLayout();
    addNoiseButton = new QPushButton("添加噪声",this);
    connect(addNoiseButton, &QPushButton::clicked, this, &MainWindow::onAddNoiseButtonClicked);
    addNoiseButton->setEnabled(false); // 初始时禁用添加噪声按钮
    noisyImageLabel = new QLabel(this);
    noisyImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    noisyImageLabel->setAlignment(Qt::AlignCenter);
    middleLayout->addWidget(addNoiseButton);
    middleLayout->addWidget(noisyImageLabel); // 将按钮添加到中间布局

    // 创建右侧布局，用于放置去噪后的图片和去噪按钮
    QVBoxLayout *rightLayout = new QVBoxLayout();
    denoiseButton = new QPushButton("图像去噪", this);
    connect(denoiseButton, &QPushButton::clicked, this, &MainWindow::onDenoiseButtonClicked);
    denoiseButton->setEnabled(false); // 初始时禁用去噪按钮
    denoisedImageLabel = new QLabel(this);
    denoisedImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    denoisedImageLabel->setAlignment(Qt::AlignCenter);
    rightLayout->addWidget(denoiseButton);
    rightLayout->addWidget(denoisedImageLabel);

    // 将左侧和右侧布局添加到主布局中
    mainLayout->addLayout(leftLayout);
    mainLayout->addLayout(middleLayout);
    mainLayout->addLayout(rightLayout);

    // 设置窗口的主布局
    setLayout(mainLayout);
}

void MainWindow::onButtonClicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("打开图片"), "", tr("图片文件 (*.png *.jpg *.bmp)"));
    originalfileName = fileName;
    if (!fileName.isEmpty()) {
        originalPixmap.load(fileName);
        originalImageLabel->setPixmap(originalPixmap.scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        noisyImageLabel->clear(); // 清除添加噪声标签
        addNoiseButton->setEnabled(true); // 启用添加噪声按钮
        denoisedImageLabel->clear(); // 清除去噪图片标签
        denoiseButton->setEnabled(true); // 启用去噪按钮
    }
}

void MainWindow::onAddNoiseButtonClicked() {
    if (originalPixmap.isNull()) {
        return;
    }

    QImage image = originalPixmap.toImage();
    cv::Mat srcImage = cv::Mat(image.height(), image.width(),
                               CV_8UC4, const_cast<uchar*>(image.bits()), static_cast<size_t>(image.bytesPerLine()));
    cv::Mat noisyImage = srcImage.clone();

    // 添加随机噪声
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 25); // 均值为0，标准差为25的正态分布

    for (int y = 0; y < noisyImage.rows; y++) {
        for (int x = 0; x < noisyImage.cols; x++) {
            for (int c = 0; c < noisyImage.channels(); c++) {  // 对于RGBA图像，需要遍历4个通道
                double noise = d(gen);;
                uchar& pixel = noisyImage.at<cv::Vec4b>(y, x)[c];
                pixel = cv::saturate_cast<uchar>(pixel + noise);
            }
        }
    }
    noisyfileName = "./pic/noise_result.png";
    cv::imwrite(noisyfileName.toStdString(), noisyImage);
    // 将带有噪声的Mat转换回QPixmap
    QImage noisyQImage = QImage(noisyImage.data, noisyImage.cols, noisyImage.rows, noisyImage.step, QImage::Format_RGBA8888).rgbSwapped();
    noisyPixmap = QPixmap::fromImage(noisyQImage);

    // 显示带有噪声的图片
    noisyImageLabel->setPixmap(noisyPixmap.scaled(noisyImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}


void MainWindow::onDenoiseButtonClicked() {
    if (noisyPixmap.isNull()) {
        return;
    }

    if(noisyPixmap.height() == noisyPixmap.width()){
        const char* resultFileName = "./pic/denoise_result.png";
        std::remove(resultFileName);
        Init();
        MainLoopDenoise(noisyfileName.toStdString());
        if(checkFileExists(resultFileName)){
            denoisedPixmap.load(QString::fromStdString(resultFileName));
            denoisedImageLabel->setPixmap(originalPixmap.scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        return;
    }

    QImage noisyImage = noisyPixmap.toImage();
    cv::Mat srcImage = cv::Mat(noisyImage.height(), noisyImage.width(),
                                CV_8UC4, const_cast<uchar*>(noisyImage.bits()), static_cast<size_t>(noisyImage.bytesPerLine()));
    cv::Mat dstImage;

    // 应用去噪算法
    cv::fastNlMeansDenoisingColored(srcImage, dstImage, 10, 10, 7, 21);

    // 将去噪后的Mat转换回QPixmap
    QImage denoisedImage = QImage(dstImage.data, dstImage.cols, dstImage.rows, dstImage.step, QImage::Format_RGBA8888).rgbSwapped();
    QPixmap denoisedPixmap = QPixmap::fromImage(denoisedImage);

    // 显示去噪后的图片
    denoisedImageLabel->setPixmap(denoisedPixmap.scaled(denoisedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    cv::imwrite("./pic/denoise_result.png", dstImage);

    // // 锐化处理
    // cv::Mat blurredImage;
    // cv::GaussianBlur(dstImage, blurredImage, cv::Size(0, 0), 3);
    // cv::addWeighted(dstImage, 1.5, blurredImage, -0.5, 0, dstImage);

    // // 将锐化后的Mat转换回QPixmap
    // QImage usmImage = QImage(dstImage.data, dstImage.cols, dstImage.rows, dstImage.step, QImage::Format_RGBA8888).rgbSwapped();
    // QPixmap usmPixmap = QPixmap::fromImage(usmImage);

    // // 显示USM锐化后的图片
    // denoisedImageLabel->setPixmap(usmPixmap.scaled(denoisedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));



}

bool checkFileExists(const std::string& filename) {
    while (true) {
        std::ifstream file(filename);
        if (file.good()) {
            std::cout << "File exists." << std::endl;
            return true;
        }

        std::cout << "File does not exist. Rechecking in 1 second..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

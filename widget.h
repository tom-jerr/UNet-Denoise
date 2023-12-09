#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <iostream>
// #include "CLDenoise.h"

class MainWindow : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void onButtonClicked();
    void onDenoiseButtonClicked();
    void onAddNoiseButtonClicked();

private:
    QLabel *originalImageLabel; // 用于显示原图的标签
    QLabel *noisyImageLabel;       // 用于显示添加噪声后的图像
    QLabel *denoisedImageLabel; // 用于显示去噪后图片的标签
    QPushButton *addNoiseButton;// 添加噪声按钮
    QPushButton *denoiseButton; // 去噪按钮
    QString originalfileName;   // 用于存储原图路径
    QString noisyfileName;   // 用于噪声图路径
    QPixmap originalPixmap;     // 用于存储原图的pixmap
    QPixmap noisyPixmap;     // 用于存储噪声图的pixmap
    QPixmap denoisedPixmap;     // 用于结果的pixmap
};
    bool checkFileExists(const std::string& filename);

#endif // WIDGET_H

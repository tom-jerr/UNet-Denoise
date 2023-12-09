QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += C:\Libtorch\libtorch\include
DEPENDPATH += C:\Libtorch\libtorch\include
LIBS += -LC:\Libtorch\libtorch\lib\
        -lc10 \
        -lc10_cuda \
        -ltorch_cpu \
        -ltorch_cuda \
        -INCLUDE:?warp_size@cuda@at@@YAHXZ \

INCLUDEPATH +=C:\Libtorch\libtorch\include\torch\csrc\api\include
DEPENDPATH += C:\Libtorch\libtorch\include\torch\csrc\api\include

INCLUDEPATH+= C:\opencv\build\include
              C:\opencv\build\include\opencv2
#             D:\opencv\opencv3.4.0\OpenCV-MinGW-Build-OpenCV-3.4.5\include\opencv2\
DEPENDPATH += C:\opencv\build\include
              C:\opencv\build\include\opencv2

LIBS+= C:/opencv/build/x64/vc15/lib/opencv_world460.lib

SOURCES += \
    CLDenoise.cpp \
    main.cpp \
    widget.cpp

HEADERS += \
    CLDenoise.h \
    CLLog.h \
    CLThreadPool.h \
    widget.h

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

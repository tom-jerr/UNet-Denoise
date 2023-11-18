import cv2 as cv
import numpy as np
#图像噪声

def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = 10000
    rows = np.random.randint(0, h, nums, dtype=np.int32)
    cols = np.random.randint(0, w, nums, dtype=np.int32)
    for i in range(nums):
        if i % 2 == 1:
            image[rows[i], cols[i]] = (255, 255, 255)
        else:
            image[rows[i], cols[i]] = (0, 0, 0)
    cv.imshow("salt pepper noise", image)
    return image

def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv.randn(noise, m, s)
    dst = cv.add(image, noise)
    cv.imshow("gaussian noise", dst)
    return dst

src = cv.imread("../test_images/unet_output_Testing.png")
h, w = src.shape[:2]
# 加入噪声
# gaussian_noise(src)
# add_salt_pepper_noise(src)
test = cv.imread("../test_images/noise_Testing.png")

result1 = cv.blur(src, (5, 5))  #图像均值
cv.imwrite("../test_images/result-1.png", result1)
result2 = cv.GaussianBlur(test, (5, 5), 0)  #高斯模糊
cv.imwrite("../test_images/result-2.png", result2)
# 能够处理较好的结果
result3 = cv.medianBlur(test, 5)  #中值滤波
cv.imshow("result-3", result3)
cv.imwrite("../test_images/result-3.png", result3)

result = cv.fastNlMeansDenoisingColored(test, None, 10, 10, 7, 21)  #非局部均值去噪
cv.imshow("result", result)
cv.imwrite("../test_images/result-4.png", result)
# 卷积核锐化
dst = np.zeros(src.shape, src.dtype)
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
# for i in range(3):
#     dst[:, :, i] = cv.filter2D(src[:, :, i], -1, kernel)
for i in range(3):
    gau = cv.GaussianBlur(src[:, :, i], (0, 0), 5)
    dst[:, :, i] = cv.addWeighted(src[:, :, i], 1.5, gau, -0.5, 0)
cv.imshow("result", dst)
cv.imwrite("../test_images/result-5.png", dst)

# Laplace锐化
for i in range(3):
    Lap = cv.Laplacian(src[:, :, i], -1)
    dst[:, :, i] = cv.addWeighted(src[:, :, i], 1, Lap, -0.5, 0)
# cv.imshow("result", dst)
cv.imwrite("../test_images/result-6.png", dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
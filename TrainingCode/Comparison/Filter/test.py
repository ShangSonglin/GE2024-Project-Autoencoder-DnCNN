import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('/Users/matsumatsu/Downloads/IMG_0214.JPG')
if image is None:
    print("Image not found")
    exit()

# 将BGR图像转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
image_gray = np.dot(image_rgb[...,:3], [0.299, 0.587, 0.114])

def add_gaussian_noise(image, mean=0, sigma=100):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def median_filter(image, filter_shape=3):
    filtered_image = image.copy()
    tmp = int(filter_shape / 2)
    height, width = image.shape
    for i in range(tmp, height - tmp + 1):
        for j in range(tmp, width - tmp + 1):
            neighborhood = image[i - tmp:i + tmp + 1, j - tmp:j + tmp + 1]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value
    return filtered_image

# 添加高斯噪声
gaussian_noise_image = add_gaussian_noise(image_gray)

# 应用中值滤波器
gaussian_median = median_filter(gaussian_noise_image)

# 显示原始图像、加噪声图像和滤波后的图像
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_noise_image, cmap='gray')
plt.title("Gaussian Noise Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_median, cmap='gray')
plt.title("Gaussian Median Filtered Image")
plt.axis('off')

plt.show()
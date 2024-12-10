import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/Users/matsumatsu/Downloads/IMG_0214.JPG')
image_array = np.array(image)
height, width, channels = image_array.shape
image_rgb = np.zeros((height, width, channels), dtype=image_array.dtype)
image_rgb[:, :, 0] = image_array[:, :, 2]
image_rgb[:, :, 1] = image_array[:, :, 1]
image_rgb[:, :, 2] = image_array[:, :, 0]
image_gray = image_rgb[:, :, 0] * 0.299 + image_rgb[:, :, 1] * 0.587 + image_rgb[:, :, 2] * 0.114
image_gray = np.round(image_gray).astype(image_array.dtype)

def add_gaussian_noise(image, mean=0, sigma=100):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def median_filter(image,filter_shape):
    filtered_image = image.copy()
    tmp = int(filter_shape/2)
    height, width = image.shape
    for i in range(tmp, height - tmp+1):
        for j in range(tmp, width - tmp+1):
            neighborhood = image[i - tmp:i + tmp+1, j - tmp:j + tmp+1]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value
    return filtered_image

gaussian_median = median_filter(gaussian_noise_image,3)

gaussian_noise_image = add_gaussian_noise(image_gray)
plt.imshow(gaussian_noise_image,cmap='gray')
plt.title("gaussian_noise")
plt.axis('off')
plt.show()

plt.imshow(gaussian_median,cmap='gray')
plt.title("Gaussian median image")
plt.axis('off')
plt.show()
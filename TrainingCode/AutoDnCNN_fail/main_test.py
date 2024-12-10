import argparse
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from skimage.io import imread, imsave
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 定义AutoEncoder模型
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 输出: 64 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出: 128 x 7 x 7
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 输出: 1 x 32 x 32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 定义DnCNN模型
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=r'C:\Users\叶聪\learngit\DnCNN-Image-Denoising\Python-Practice\testsets', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['NoisyImages'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('../models', 'DnCNN_sigma25'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),  # 使用较小的目标分辨率
        transforms.ToTensor()
    ])
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # 增加 Batch 维度
    return img

def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)  # 限制像素值范围在[0,1]

def denoise_images(args, autoencoder, dncnn, device):
    for set_cur in args.set_names:
        set_dir = os.path.join(args.set_dir, set_cur)
        if not os.path.exists(set_dir):
            print(f'Set directory not found: {set_dir}')
            continue

        result_set_dir = os.path.join(args.result_dir, set_cur)
        os.makedirs(result_set_dir, exist_ok=True)
        print(f'Saving results to: {result_set_dir}')

        psnrs = []
        ssims = []

        for im in os.listdir(set_dir):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # 读取图像并添加噪声
                x = np.array(imread(os.path.join(set_dir, im)), dtype=np.float32) / 255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, 1, y.shape[0], y.shape[1]).to(device)

                # 使用 AutoEncoder 进行初步去噪
                with torch.no_grad():
                    x_autoencoded = autoencoder(y_)
                    x_autoencoded = x_autoencoded.view(y.shape[0], y.shape[1])
                    x_autoencoded = x_autoencoded.cpu().numpy().astype(np.float32)

                # 使用 DnCNN 进行进一步去噪
                with torch.no_grad():
                    x_dncnn = dncnn(torch.from_numpy(x_autoencoded).view(1, 1, x_autoencoded.shape[0], x_autoencoded.shape[1]).to(device))
                    x_dncnn = x_dncnn.view(x_autoencoded.shape[0], x_autoencoded.shape[1])
                    x_dncnn = x_dncnn.cpu().numpy().astype(np.float32)

                psnr_x_ = compare_psnr(x, x_dncnn, data_range=1.0)
                ssim_x_ = compare_ssim(x, x_dncnn, data_range=1.0)
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

                if args.save_result:
                    name, ext = os.path.splitext(im)
                    result_path = os.path.join(result_set_dir, name + '_denoised' + ext)
                    imsave(result_path, np.clip(x_dncnn, 0, 1))  # save the denoised image
                    print(f'Saved denoised image: {result_path}')

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        print(f'Dataset: {set_cur} \n  PSNR = {psnr_avg:.2f} dB, SSIM = {ssim_avg:.4f}')

        if args.save_result:
            results_path = os.path.join(result_set_dir, 'results.txt')
            np.savetxt(results_path, np.hstack((psnrs, ssims)), fmt='%2.4f')
            print(f'Saved results: {results_path}')

import torch.serialization

# 添加 DnCNN 到安全全局变量列表
torch.serialization.add_safe_globals([DnCNN])

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载AutoEncoder模型
    autoencoder = ConvAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load('phase1/autoencoder_weights.pth', map_location=device))
    autoencoder.eval()

    # 加载DnCNN模型
    dncnn = DnCNN().to(device)
    dncnn.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name), map_location=device))
    dncnn.eval()

    # 调用 denoise_images 函数
    denoise_images(args, autoencoder, dncnn, device)
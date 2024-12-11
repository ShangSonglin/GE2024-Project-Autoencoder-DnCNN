import argparse
import os
import time
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from skimage.io import imread, imsave
from skimage.color import rgb2gray

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=r'C:\Users\叶聪\learngit\DnCNN\testsets', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['NoisyImages'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('../../AutoDnCNN_remake/models', 'DnCNN_sigma25'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        # 将浮点数图像转换为 8 位整数图像
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        imsave(path, result)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

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
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__ == '__main__':
    args = parse_args()

    image_dir = args.set_dir  # 使用命令行中传入的 --set_dir 参数
    result_dir = args.result_dir  # 使用命令行中传入的 --result_dir 参数

    os.makedirs(result_dir, exist_ok=True)  # 如果结果目录不存在，则创建
    # 加载模型
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_path):
        log(f'Model file not found: {model_path}')
        exit(1)
    model = torch.load(model_path, map_location=torch.device('cpu'))  # 确保模型在 CPU 上加载
    log('Model loaded successfully')

    model.eval()  # evaluation mode

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        set_dir = os.path.join(args.set_dir, set_cur)
        if not os.path.exists(set_dir):
            log(f'Set directory not found: {set_dir}')
            continue

        result_set_dir = os.path.join(args.result_dir, set_cur)
        os.makedirs(result_set_dir, exist_ok=True)
        log(f'Saving results to: {result_set_dir}')

        psnrs = []
        ssims = []

        for im in os.listdir(set_dir):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                img_path = os.path.join(set_dir, im)
                log(f'Processing image: {img_path}')
                x = np.array(imread(img_path), dtype=np.float32) / 255.0
                if x.ndim == 3:
                    x = rgb2gray(x)  # 将彩色图像转换为灰度图像
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, 1, y.shape[0], y.shape[1])  # 确保输入有1个通道

                start_time = time.time()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.detach().numpy().astype(np.float32)
                elapsed_time = time.time() - start_time
                log(f'{set_cur} : {im} : {elapsed_time:.4f} seconds')

                psnr_x_ = compare_psnr(x, x_, data_range=1.0)
                ssim_x_ = compare_ssim(x, x_, data_range=1.0)
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

                if args.save_result:
                    name, ext = os.path.splitext(im)
                    result_path = os.path.join(result_set_dir, name + '_dncnn' + ext)
                    save_result(x_, path=result_path)  # save the denoised image
                    log(f'Saved denoised image: {result_path}')

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        log(f'Dataset: {set_cur} \n  PSNR = {psnr_avg:.2f} dB, SSIM = {ssim_avg:.4f}')

        if args.save_result:
            results_path = os.path.join(result_set_dir, 'results.txt')
            save_result(np.hstack((psnrs, ssims)), path=results_path)
            log(f'Saved results: {results_path}')
import torch
import os
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 确保导入模型定义
from TrainingCode.AutoDnCNN_remake.train.main_train import ConvAutoencoder, DnCNN


def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def denoise_image(autoencoder_model, dncnn_model, image):
    # 首先使用自动编码器去噪
    autoencoder_output = autoencoder_model(image)

    # 然后使用DnCNN去噪
    dncnn_output = dncnn_model(autoencoder_output)

    return dncnn_output.squeeze()


def save_result(result, path):
    result = np.clip(result, 0, 1)
    Image.fromarray((result * 255).astype(np.uint8)).save(path)


def log(*args, **kwargs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=r'C:\Users\叶聪\learngit\AutoDnCNN_Image_Denoising\TrainingCode\models',
                        help='directory of the model')
    parser.add_argument('--model_name_autoencoder', default='autoencoder_weights.pth', type=str, help='the model name')
    parser.add_argument('--model_name_dncnn', default='dncnn_weights.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)  # 如果结果目录不存在，则创建

    # 加载模型
    autoencoder_model = load_model(os.path.join(args.model_dir, args.model_name_autoencoder), ConvAutoencoder)
    dncnn_model = load_model(os.path.join(args.model_dir, args.model_name_dncnn), DnCNN)

    # 加载MNIST测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 测试模型
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= 10:  # 只处理前10张图片
            break
        log(f'Processing image {batch_idx + 1}/{len(test_loader)}')
        noisy_images = data.squeeze()  # 获取带噪声的图片
        denoised_images = denoise_image(autoencoder_model, dncnn_model, data)

        # 保存原始带噪声的图片
        save_result(noisy_images.detach().numpy(), os.path.join(args.result_dir, f'noisy_{batch_idx}.png'))
        log(f'Saved noisy image: noisy_{batch_idx}.png')

        # 保存去噪后的图片
        save_result(denoised_images.detach().numpy(), os.path.join(args.result_dir, f'denoised_{batch_idx}.png'))
        log(f'Saved denoised image: denoised_{batch_idx}.png')
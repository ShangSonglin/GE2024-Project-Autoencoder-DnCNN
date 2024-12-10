import torch
import os
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as compare_ssim
from TrainingCode.AutoDnCNN_remake.train.main_train import ConvAutoencoder, DnCNN

def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def denoise_image(model, image):
    # 添加噪声
    noisy_image = image + torch.randn_like(image) * 0.3
    noisy_image = torch.clamp(noisy_image, 0., 1.)

    # 去噪
    with torch.no_grad():
        denoised_image = model(noisy_image)

    return denoised_image.squeeze()

def save_result(result, path):
    result = np.clip(result, 0, 1)  # 确保值在[0, 1]范围内
    result = (result * 255).astype(np.uint8)  # 映射到[0, 255]范围并转换为整数
    Image.fromarray(result).save(path)  # 保存图像

def log(*args, **kwargs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=r'C:\Users\叶聪\learngit\AutoDnCNN_Image_Denoising\testsets', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['NoisyImages'], help='directory of test dataset')
    parser.add_argument('--model_dir', default=r'C:\Users\叶聪\learngit\AutoDnCNN_Image_Denoising\TrainingCode\models', help='directory of the model')
    parser.add_argument('--model_name_autoencoder', default='autoencoder_weights.pth', type=str, help='the model name')
    parser.add_argument('--model_name_dncnn', default='dncnn_weights.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)  # 如果结果目录不存在，则创建

    # 加载模型
    autoencoder_model = load_model(os.path.join(args.model_dir, args.model_name_autoencoder), ConvAutoencoder)
    dncnn_model = load_model(os.path.join(args.model_dir, args.model_name_dncnn), DnCNN)

    for set_name in args.set_names:
        set_dir = os.path.join(args.set_dir, set_name)
        result_set_dir = os.path.join(args.result_dir, set_name)
        os.makedirs(result_set_dir, exist_ok=True)
        log(f'Saving results to: {result_set_dir}')

        for im_name in os.listdir(set_dir):
            if im_name.endswith(".jpg") or im_name.endswith(".bmp") or im_name.endswith(".png"):
                img_path = os.path.join(set_dir, im_name)
                log(f'Processing image: {img_path}')
                image = Image.open(img_path).convert('L')  # 转换为灰度图
                image = transforms.ToTensor()(image).unsqueeze(0)  # 增加批次维度

                denoised_image_autoencoder = denoise_image(autoencoder_model, image)
                denoised_image_dncnn = denoise_image(dncnn_model, image)

                save_result(denoised_image_autoencoder.squeeze().numpy(), os.path.join(result_set_dir, f'{os.path.splitext(im_name)[0]}_autoencoder.png'))
                save_result(denoised_image_dncnn.squeeze().numpy(), os.path.join(result_set_dir, f'{os.path.splitext(im_name)[0]}_dncnn.png'))
                log(f'Saved denoised images: {os.path.splitext(im_name)[0]}')
import torch
import os
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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
    def save_result(result, path):
        result = np.clip(result, 0, 1)
        Image.fromarray((result * 255).astype(np.uint8)).save(path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default=r'C:\Users\叶聪\learngit\AutoDnCNN_Image_Denoising\testsets', type=str,
                        help='directory of test dataset')
    parser.add_argument('--set_names', default=['BSD68', 'NoisyImages', 'Set12'], help='directory of test dataset')
    parser.add_argument('--model_dir', default=r'C:\Users\叶聪\learngit\AutoDnCNN_Image_Denoising\TrainingCode\models',
                        help='directory of the model')
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

                # 去噪
                denoised_image = denoise_image(autoencoder_model, dncnn_model, image)

                # 保存去噪后的图像
                if args.save_result:
                    save_result(denoised_image.detach().squeeze().numpy(),
                                os.path.join(result_set_dir, f'{os.path.splitext(im_name)[0]}_denoised.png'))
                    log(f'Saved denoised image: {os.path.splitext(im_name)[0]}_denoised.png')
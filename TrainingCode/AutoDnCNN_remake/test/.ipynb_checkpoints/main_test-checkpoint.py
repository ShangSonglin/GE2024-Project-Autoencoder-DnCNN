import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from TrainingCode.AutoDnCNN_remake.train.main_train import ConvAutoencoder, DnCNN

def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def denoise_image(model, image_path):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 添加噪声
    noisy_image = image + torch.randn_like(image) * 0.3
    noisy_image = torch.clamp(noisy_image, 0., 1.)

    # 去噪
    with torch.no_grad():
        denoised_image = model(noisy_image)

    return denoised_image.squeeze().numpy()

if __name__ == '__main__':
    autoencoder_model = load_model('autoencoder_weights.pth', ConvAutoencoder)
    dncnn_model = load_model('dncnn_weights.pth', DnCNN)

    image_path = 'path_to_your_noisy_image.jpg'  # 指定你的噪声图像路径
    denoised_image = denoise_image(autoencoder_model, image_path)
    # 保存或显示去噪后的图像
    # ...
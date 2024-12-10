import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    # def __init__(self):
    #     super(ConvAutoencoder, self).__init__()
    #     # 编码器
    #     self.encoder = nn.Sequential(
    #         nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(),
    #         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(512),
    #         nn.ReLU()
    #     )
    #     # 解码器
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
    #         nn.Sigmoid()
    #     )
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
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: 1 x 32 x 32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 数据预处理：添加噪声
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x + 0.5 * torch.randn_like(x), 0., 1.))  # 添加噪声后使用torch.clamp限制值域
])

# 数据预处理：调整为 32x32
transform = transforms.Compose([
    transforms.Grayscale(),  # 转为单通道灰度图像
    transforms.Resize((32, 32)),  # 将输入调整为 32x32
    transforms.ToTensor()  # 转为张量
])
# 加载 MNIST 数据集
train_dataset = datasets.MNIST(
    root='./data',  # 数据存储路径
    train=True,  # 训练数据
    download=True,  # 如果数据不存在，下载数据
    transform=transform  # 应用预处理
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # 每个批次的数据量
    shuffle=True  # 打乱数据
)
# 数据预处理
from torchvision import transforms
from PIL import Image


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),  # 使用较小的目标分辨率
        transforms.ToTensor()
    ])
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # 增加 Batch 维度
    return img


# # 准备数据集
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: torch.clamp(x + 0.5 * torch.randn_like(x), 0., 1.))  # 添加噪声后使用torch.clamp限制值域
# ])

# # 数据预处理：调整为 32x32
# transform = transforms.Compose([
#     transforms.Grayscale(),  # 转为单通道灰度图像
#     transforms.Resize((32, 32)),  # 将输入调整为 32x32
#     transforms.ToTensor()  # 转为张量
# ])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = ConvAutoencoder()
outputs = model(noisy_images)
print("输入图像形状:", noisy_image.shape)
print("输出图像形状:", outputs.shape)
print("目标图像形状:", clean_images.shape)


# 初始化模型

# 初始化权重
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# model.apply(weights_init)
import torch.nn.functional as F

# 动态调整目标图像的分辨率
if clean_images.shape != outputs.shape:
    clean_images_resized = F.interpolate(clean_images, size=outputs.shape[2:])
else:
    clean_images_resized = clean_images
# 损失函数
criterion = nn.MSELoss()
# 计算损失
loss = criterion(outputs, clean_images_resized)
print("输出图像形状:", outputs.shape)
print("目标图像形状:", clean_images.shape)

# 初始化优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 示例调用
image_path = '/Users/matsumatsu/Downloads/IMG_0214.JPG'
preprocessed_img = preprocess_image(image_path)
print("预处理后的图片形状:", preprocessed_img.shape)

num_epochs = 5
for epoch in range(num_epochs):
    for batch in train_loader:
        clean_images, _ = batch
        noisy_images = clean_images + torch.randn_like(clean_images) * 0.3  # 添加噪声
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # 前向传播
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # 调整学习率
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存权重
torch.save(model.state_dict(), 'autoencoder_weights.pth')
print("模型权重已保存为 'autoencoder_weights.pth'")


def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)  # 限制像素值范围在[0,1]


# 可视化图片
def show_images(original, noisy, denoised):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original.squeeze(0).squeeze(0), cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(noisy.squeeze(0).squeeze(0), cmap='gray')
    axs[1].set_title("Noisy Image")
    axs[1].axis('off')

    axs[2].imshow(denoised.squeeze(0).squeeze(0).detach().numpy(), cmap='gray')
    axs[2].set_title("Denoised Image")
    axs[2].axis('off')

    plt.show()


# 替换为 PNG 图片路径
image_path = '/Users/matsumatsu/Downloads/IMG_0214.JPG'

# 预处理图片
original_image = preprocess_image(image_path)  # 加载原始图片

# 添加噪声
noisy_image = add_noise(original_image)  # 添加噪声

# 使用模型去噪
denoised_image = model(noisy_image)  # 使用训练好的模型进行去噪

# 显示结果
show_images(original_image, noisy_image, denoised_image)

print("Noisy Image Min:", noisy_images.min(), "Max:", noisy_images.max())
print("Denoised Image Min:", outputs.min(), "Max:", outputs.max())
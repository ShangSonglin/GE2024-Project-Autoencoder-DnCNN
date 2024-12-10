import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 定义ConvAutoencoder
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

# 定义DnCNN（假设结构与ConvAutoencoder相同）
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out


# 加载数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化模型
autoencoder_model = ConvAutoencoder()
dncnn_model = DnCNN()

# 初始化优化器
optimizer_autoencoder = optim.Adam(autoencoder_model.parameters(), lr=0.001)
optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=0.001)

# 损失函数
criterion = nn.MSELoss()

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        clean_images, _ = batch
        noisy_images = clean_images + torch.randn_like(clean_images) * 0.3  # 添加噪声
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # 自动编码器去噪
        optimizer_autoencoder.zero_grad()
        autoencoder_outputs = autoencoder_model(noisy_images)
        loss_autoencoder = criterion(autoencoder_outputs, clean_images)
        loss_autoencoder.backward()
        optimizer_autoencoder.step()

        # DnCNN去噪
        optimizer_dncnn.zero_grad()
        dncnn_outputs = dncnn_model(noisy_images)
        loss_dncnn = criterion(dncnn_outputs, clean_images)
        loss_dncnn.backward()
        optimizer_dncnn.step()

    # 打印每个epoch的损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Autoencoder Loss: {loss_autoencoder.item():.4f}, DnCNN Loss: {loss_dncnn.item():.4f}")

# 保存权重
torch.save(autoencoder_model.state_dict(), 'autoencoder_weights.pth')
torch.save(dncnn_model.state_dict(), 'dncnn_weights.pth')
print("模型权重已保存为 'autoencoder_weights.pth' 和 'dncnn_weights.pth'")
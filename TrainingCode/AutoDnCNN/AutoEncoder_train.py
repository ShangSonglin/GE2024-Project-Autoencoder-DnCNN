import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# 初始化模型
model = ConvAutoencoder()

# 损失函数
criterion = nn.MSELoss()

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for batch in train_loader:
        images, _ = batch
        noisy_images = images + torch.randn_like(images) * 0.3  # 添加噪声
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # 前向传播
        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存权重
torch.save(model.state_dict(), 'autoencoder_weights.pth')

print("模型权重已保存为 'autoencoder_weights.pth'")
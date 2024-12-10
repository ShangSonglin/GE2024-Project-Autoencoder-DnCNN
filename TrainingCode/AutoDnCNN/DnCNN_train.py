import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义 DnCNN 模型
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = kernel_size // 2
        layers = []

        # 第一层
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)


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
model = DnCNN()

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

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存权重
torch.save(model.state_dict(), 'dncnn_weights.pth')

print("模型权重已保存为 'dncnn_weights.pth'")

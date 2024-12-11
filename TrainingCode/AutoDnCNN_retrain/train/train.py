import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.init as init

# 定义ConvAutoencoder和DnCNN模型
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = [
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(n_channels, eps=1e-4, momentum=0.95),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return x - self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

# 检测 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    return model

def train_and_test_models():
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # 数据预处理和加载数据集
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")

    # 初始化模型
    autoencoder_model = ConvAutoencoder().to(device)
    dncnn_model = DnCNN(depth=35, n_channels=128).to(device)

    # 如果存在权重文件则加载
    autoencoder_model = load_model(autoencoder_model, 'autoencoder_weights.pth')
    dncnn_model = load_model(dncnn_model, 'dncnn_weights.pth')

    # 优化器和调度器
    optimizer_autoencoder = optim.Adam(autoencoder_model.parameters(), lr=learning_rate)
    optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=learning_rate)
    scheduler_autoencoder = optim.lr_scheduler.StepLR(optimizer_autoencoder, step_size=5, gamma=0.5)
    scheduler_dncnn = optim.lr_scheduler.StepLR(optimizer_dncnn, step_size=5, gamma=0.5)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        autoencoder_model.train()
        dncnn_model.train()
        total_autoencoder_loss = 0.0
        total_dncnn_loss = 0.0

        for batch_idx, (clean_images, _) in enumerate(dataloader):
            clean_images = clean_images.to(device)
            noisy_images = clean_images + torch.randn_like(clean_images) * 0.3
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            # 自动编码器去噪
            optimizer_autoencoder.zero_grad()
            autoencoder_outputs = autoencoder_model(noisy_images)
            loss_autoencoder = criterion(autoencoder_outputs, clean_images)
            loss_autoencoder.backward()
            optimizer_autoencoder.step()
            total_autoencoder_loss += loss_autoencoder.item()

            # DnCNN去噪
            optimizer_dncnn.zero_grad()
            dncnn_outputs = dncnn_model(noisy_images)
            loss_dncnn = criterion(dncnn_outputs, clean_images)
            loss_dncnn.backward()
            optimizer_dncnn.step()
            total_dncnn_loss += loss_dncnn.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}] ")
                print(f"Autoencoder Loss: {loss_autoencoder.item():.4f}, DnCNN Loss: {loss_dncnn.item():.4f}")

        scheduler_autoencoder.step()
        scheduler_dncnn.step()

        # 保存模型权重
        save_model(autoencoder_model, 'autoencoder_weights.pth')
        save_model(dncnn_model, 'dncnn_weights.pth')

        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed | Autoencoder Avg Loss: {total_autoencoder_loss / len(dataloader):.4f} | DnCNN Avg Loss: {total_dncnn_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    train_and_test_models()

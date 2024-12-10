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

def train_and_test_models():
    num_epochs = 1
    batch_size = 64
    learning_rate = 0.001

    # 数据预处理和加载数据集
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    autoencoder_model = ConvAutoencoder()
    dncnn_model = DnCNN()

    # 初始化优化器
    optimizer_autoencoder = optim.Adam(autoencoder_model.parameters(), lr=learning_rate)
    optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            clean_images, _ = batch
            noisy_images = clean_images + torch.randn_like(clean_images) * 0.3
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
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Autoencoder Loss: {loss_autoencoder.item():.4f}, DnCNN Loss: {loss_dncnn.item():.4f}")

    # 保存权重
    torch.save(autoencoder_model.state_dict(), 'autoencoder_weights.pth')
    torch.save(dncnn_model.state_dict(), 'dncnn_weights.pth')
    print("模型权重已保存为 'autoencoder_weights.pth' 和 'dncnn_weights.pth'")


if __name__ == '__main__':
    train_and_test_models()

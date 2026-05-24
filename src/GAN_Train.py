import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
latent_dim = 100  # 随机噪声的维度
batch_size = 64
epochs = 20
learning_rate = 0.0002
beta1 = 0.5  # Adam优化器的参数


# 生成器：根据数据集的大小和通道数进行调整
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, img_size):
        super(Generator, self).__init__()

        # 保存channels和img_size为实例变量
        self.channels = channels
        self.img_size = img_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.channels * self.img_size * self.img_size),  # 改为 channels * img_size * img_size
            nn.Tanh()  # 输出图像为[0,1]之间的值
        )

    def forward(self, z):
        return self.fc(z).view(-1, self.channels * self.img_size * self.img_size)  # 输出图像大小为[channels, img_size, img_size]


# 判别器：根据数据集的大小和通道数进行调整
class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出0或1的概率
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))  # 将图像展平输入网络


# 训练函数
def train_gan(generator, discriminator, dataset, epochs=20, latent_dim=100, batch_size=64, device=device):
    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # 损失函数
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # 标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            discriminator.zero_grad()

            # 真实图像的损失
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # 生成图像并计算损失
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            generator.zero_grad()

            # 计算生成图像的损失
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)  # 生成器想要判别器认为生成的图像是真实的
            g_loss.backward()
            optimizer_g.step()

            if i % 200 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Step [{i}/{len(data_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

    # 每个epoch结束时保存模型
    torch.save(generator.state_dict(), 'ImageNet_generator.pth')
    torch.save(discriminator.state_dict(), 'ImageNet_discriminator.pth')


# 数据加载函数：根据数据集选择合适的图像大小和通道数
def get_transform(dataset_name):
    if dataset_name == "FashionMNIST":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    elif dataset_name == "CIFAR10":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif dataset_name == "GTSRB":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif dataset_name == "ImageNet":
        return transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def get_img_size_and_channels(dataset_name):
    if dataset_name == "FashionMNIST":
        return 28, 1  # 28x28, 灰度图
    elif dataset_name == "CIFAR10":
        return 32, 3  # 32x32, RGB图像
    elif dataset_name == "GTSRB":
        return 32, 3  # 32x32, RGB图像
    elif dataset_name == "ImageNet":
        return 64, 3  # 64x64, RGB图像

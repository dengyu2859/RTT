import torch
import os
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms




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

def generate_and_save_images(generator, latent_dim, num_images, output_folder, device, img_size, channels):
    # 确保保存的文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 生成图像
    generator.eval()  # 设置生成器为评估模式
    with torch.no_grad():  # 不需要计算梯度
        for i in range(num_images):
            # 生成随机噪声
            noise = torch.randn(1, latent_dim).to(device)

            # 生成图像
            fake_image = generator(noise)
            fake_image = fake_image.view(1, 3, 64, 64)

            # 保存图像
            save_image(fake_image, os.path.join(output_folder, f"generated_image_{i + 1}.png"))

    print(f"{num_images} images have been saved to {output_folder}.")


def get_img_size_and_channels(dataset_name):
    if dataset_name == "FashionMNIST":
        return 28, 1  # 28x28, 灰度图
    elif dataset_name == "CIFAR10":
        return 32, 3  # 32x32, RGB图像
    elif dataset_name == "GTSRB":
        return 32, 3  # 32x32, RGB图像
    elif dataset_name == "ImageNet":
        return 64, 3  # 64x64, RGB图像



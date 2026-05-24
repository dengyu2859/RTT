import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pytorch_pretrained_biggan import BigGAN

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型 (例如 biggan-deep-256)
model = BigGAN.from_pretrained('biggan-deep-256')
model.to(device)
model.eval()
print("BigGAN model loaded successfully!")

# 数据集配置
DATASET_CONFIGS = {
    'fashion_mnist': {
        'size': 28,
        'channels': 1,  # 灰度图
        'transform': transforms.Compose([transforms.Resize(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    },
    'cifar10': {
        'size': 32,
        'channels': 3,  # RGB图
        'transform': transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    'gtsrb': {
        'size': 32,
        'channels': 3,  # RGB图
        'transform': transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    'tiny_imagenet': {
        'size': 64,
        'channels': 3,  # RGB图
        'transform': transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

# 生成图像的函数
def generate_images(dataset_name, num_images=2000):
    latent_dim = 128  # BigGAN的潜在空间维度

    # 创建保存图像的文件夹
    save_dir = f"generated_images_{dataset_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(1, num_images + 1):
        print(f"Generating image {i} of {num_images}...")

        # 为当前图像生成随机噪声
        z = torch.randn(1, latent_dim).to(device)

        # BigGAN生成的类别标签，BigGAN是基于ImageNet训练的，生成图像的类别在0到999之间
        class_vector = torch.randint(0, 1000, (1,)).to(device)  # 随机选择类别
        class_vector = torch.nn.functional.one_hot(class_vector, num_classes=1000).float().to(device)

        # 使用BigGAN生成图像
        with torch.no_grad():
            output = model(z, class_vector, truncation=0.4)  # truncation值控制生成图像的质量

        # 获取目标数据集的大小和通道数
        dataset_config = DATASET_CONFIGS[dataset_name]
        target_size = dataset_config['size']
        target_channels = dataset_config['channels']

        # 对生成的图像进行调整并保存
        output_image = output.squeeze(0).cpu().numpy()
        output_image = np.transpose(output_image, (1, 2, 0))  # 转置为(H, W, C)

        # 如果生成的图像通道数不符合要求，则转换
        if target_channels == 1:  # 如果是灰度图（如FashionMNIST）
            output_image = np.dot(output_image[...,:3], [0.2989, 0.587, 0.114])  # 转为灰度图
            output_image = np.expand_dims(output_image, axis=-1)  # 添加通道维度
        else:  # 否则，保持RGB格式
            output_image = np.clip(output_image, 0, 1)  # 确保图像在[0,1]之间

        # 调整图像尺寸
        pil_image = Image.fromarray((output_image * 255).astype(np.uint8))  # 转回PIL格式

        # 根据数据集要求调整图像大小
        pil_image = pil_image.resize((target_size, target_size))

        # 保存图像
        image_filename = os.path.join(save_dir, f"generated_image_{i}.png")
        pil_image.save(image_filename)

    print(f"Generated {num_images} images in {save_dir}.")

# 调用生成图像函数
dataset_name = 'fashion_mnist'  # 可以修改为'fashion_mnist', 'cifar10', 'gtsrb', 'tiny_imagenet'
generate_images(dataset_name, num_images=2000)

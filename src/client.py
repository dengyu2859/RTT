import copy
import numpy as np
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt
import utils
from model.CNN import CNN_layer2, CNN_layer3
from torch.utils.data import DataLoader
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class Client():
    def __init__(self, _id, args, loss_func, model_type, poisoned_indices, client_datasets, S, K, alpha):
        self.K, self.S, self.alpha = K, S, alpha
        self.normal_dataset = client_datasets
        self.poisoned_indices = poisoned_indices
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.n_data = len(self.normal_dataset)
        self.model, self.hook = utils.model_choice(model_type, args)
        self.is_tabular = (args.dataset in ['LOAN', 'AGNews', 'DBPedia'])
        if not self.is_tabular:
            if self.args.dataset == 'FashionMNIST':
                self.dist_model = DomainDiscriminator(in_channels=1).to(self.args.device)
            else:
                self.dist_model = DomainDiscriminator(in_channels=3).to(self.args.device)
            self.dist_model_loss_function = nn.BCELoss()
        # # 根据传入的 model_type 参数选择模型
        # if model_type == 'CNN2':
        #     self.model = CNN_layer2(args).to(args.device)
        # elif model_type == 'CNN3':
        #     self.model = CNN_layer3(args).to(args.device)

    def local_train(self, test_dataset, verbose=False):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)
        acc_test, back_acc = 0.0, 0.0

        for epoch_idx in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)     # 计算损失函数
                loss.backward()                       # 反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()                      # 更新模型参数

            if verbose:
                with torch.no_grad():
                    if self.is_tabular and not self.args.attack:
                        acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                        back_acc, back_loss = 0.0, 0.0
                    else:
                        acc_test, acc_loss, back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S,
                                                                               self.alpha, self.args)
                    print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.2f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.2f}")
                self.model.train()
        return acc_test, back_acc

    # 客户端推理 logits
    def local_predict_logits(self, distill_dataset, epoch, verbose=False):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            all_logits = []
            for images, _ in test_loader:
                inputs = images.to(device=self.args.device)
                outputs = self.model(inputs)
                all_logits.append(outputs.cpu() if isinstance(outputs, torch.Tensor) else outputs.cpu())

        all_logits = torch.cat(all_logits, dim=0)

        return all_logits

    # 客户端蒸馏
    def local_distill(self, distill_dataset, avg_logits, epoch, verbose=False):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.distill_lr, momentum=self.args.distill_momentum)

        # 仅为蒸馏数据集创建 DataLoader，并关闭打乱功能
        distill_loader = DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)

        # 将 avg_logits 移动到正确的设备，并禁用梯度计算
        avg_logits = avg_logits.to(self.args.device, non_blocking=True)

        for epoch_idx in range(self.args.distill_ep):
            epoch_loss = 0.0  # 用于累加每个 epoch 的总损失
            # 使用 enumerate() 来获取批次索引
            for i, (images, _) in enumerate(distill_loader):
                optimizer.zero_grad()

                # 手动从 avg_logits 中提取对应的软标签
                # 这里的切片依赖于 DataLoader 不打乱数据的特性
                start_idx = i * self.args.local_bs
                end_idx = start_idx + images.size(0)  # 使用 images.size(0) 确保最后一个不满批次的批次也能正确切片
                teacher_logits = avg_logits[start_idx:end_idx]

                # 将数据移动到设备
                inputs = images.to(self.args.device, non_blocking=True)

                # 学生模型前向传播
                student_logits = self.model(inputs)

                # 计算蒸馏损失
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / self.args.temperature, dim=1),
                    F.softmax(teacher_logits / self.args.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.args.temperature ** 2)

                total_loss = distill_loss

                distill_loss.backward()
                optimizer.step()

                # 累加每个 batch 的损失
                epoch_loss += total_loss.item() * inputs.size(0)

            if verbose:
                avg_epoch_loss = epoch_loss / len(distill_loader.dataset)
                print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| distill: {avg_epoch_loss:.3f}")

        # 图片可视化
    def Show_img(self, tensor_img, label=None, mean=(0.5,), std=(0.5,)):
        # 反归一化
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        img = inv_normalize(tensor_img.cpu())
        img = torch.clamp(img, 0, 1)  # 保证范围在 [0,1]

        # 显示
        plt.imshow(img.permute(1, 2, 0))  # C,H,W → H,W,C
        if label is not None:
            plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    # 增加一个领域训练函数
    def Domain_train(self, test_dataset):
        if self.is_tabular:
            return
        Positive_dataset = copy.deepcopy(self.normal_dataset)
        gauss_dataset = copy.deepcopy(test_dataset)
        pos_dataset = DomainDataset(Positive_dataset, domain_label=1.0)
        neg_dataset = DomainDataset(gauss_dataset, domain_label=0.0)
        combined_dataset = ConcatDataset([pos_dataset, neg_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
        self.dist_model.train()
        optimizer = torch.optim.Adam(self.dist_model.parameters(), lr=1e-3)

        for epoch_idx in range(20):
            for _, (images, labels) in enumerate(combined_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                labels = labels.view(-1, 1)
                outputs = self.dist_model(inputs)
                loss = self.dist_model_loss_function(outputs, labels)     # 计算损失函数
                loss.backward()                       # 反向传播
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()                      # 更新模型参数

        del combined_loader, combined_dataset, pos_dataset, neg_dataset, Positive_dataset, gauss_dataset

    def get_domain_score(self, distill_dataset):
        if self.is_tabular:
            return torch.zeros(len(distill_dataset), device=self.args.device)
        self.dist_model.eval()
        self.dist_model.to(self.args.device)
        all_score = []
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            for images, _ in test_loader:

                images = images.to(self.args.device)
                scores = self.dist_model(images)
                all_score.append(scores.cpu())

        all_scores = torch.cat(all_score, dim=0)
        return all_scores


# 领域鉴别器：专攻“这是不是我见过的图片
class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(DomainDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 无论输入多大，都池化成 4x4 的特征图
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        # 此时全连接层输入固定为 32 * 4 * 4 = 512
        self.disc_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc_head(self.features(x))


class DomainDataset(Dataset):
    def __init__(self, original_dataset, domain_label):
        """
        original_dataset: 原始的 PyTorch 数据集 (如 MNIST, CIFAR10 等)
        domain_label: 统一修改后的标签 (1 代表正样本，0 代表负样本)
        """
        self.original_dataset = original_dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始的图片和原始类别标签
        image, _ = self.original_dataset[idx]

        # 忽略原始标签，直接返回图像和我们定义的领域标签 (0 或 1)
        # 注意：鉴别器训练通常需要 float 类型的标签用于 BCELoss
        return image, torch.tensor(self.domain_label, dtype=torch.float32)
import copy
import torch.nn as nn
import torch
import utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.CNN import CNN_layer2, CNN_layer3
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# 恶意客户端
class Malicious_client():
    def __init__(self, _id, args, loss_func, model_type, poisoned_indices, client_datasets):
        self.desired_margin = 50  # 初始目标差距
        self.normal_dataset = client_datasets
        self.poisoned_indices = poisoned_indices
        self.poisoning_proportion = 0.6
        self.global_logits = None
        self.up_margin_step = 20
        self.down_margin_step = 25
        self.factor = 3
        self.adjustment_rate = 0.1
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        # 根据传入的 model_type 参数选择模型
        self.model, self.hook = utils.model_choice(model_type, args)
        self.input_shape = self.normal_dataset[0][0].shape
        # self.get_tigger()
        self.y_t = self.args.back_target  # 后门目标类，假设为 6 (Bag)
        self.S = args.S  # 触发区域大小
        if args.dataset == 'FashionMNIST':
            self.K = torch.randn(self.S, self.S)
        elif args.dataset in ['CIFAR10', 'CIFAR100']:
            self.K = torch.randn(3, self.S, self.S)
        elif args.dataset in ['ImageNet', 'GTSRB']:
            self.K = torch.randn(3, self.S, self.S)
        self.BETA = args.beta  # 污染比例
        self.local_poisoned_indices, self.alpha = self.get_poisoned_indices()
        self.poisoned_trainset = PoisonedDataset(self.normal_dataset, self.local_poisoned_indices, self.args.back_target)
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=False, drop_last=True)

        if self.args.dataset == 'FashionMNIST':
            self.dist_model = DomainDiscriminator(in_channels=1).to(self.args.device)
        else:
            self.dist_model = DomainDiscriminator(in_channels=3).to(self.args.device)
        self.dist_model_loss_function = nn.BCELoss()

        # 对比实验使用
        self.mask, self.pattern = utils.build_adaptive_trigger(self.input_shape, self.args)

    def return_params(self):
        return self.S, self.K, self.alpha

    def local_train(self, test_dataset, verbose=False):
        # 加载新的全局模型
        self.model.train()

        if self.args.attack_type == 'SADBA':
            self.target_center = self.get_target_feature_center()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for epoch_idx in range(self.args.local_ep):
            for i, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                if self.args.attack_type == 'DBA':
                    # self.Show_img(inputs[1], labels[1])
                    self.Implant_trigger(inputs, labels)
                    # self.Show_img(inputs[1], labels[1])
                if self.args.attack_type == 'SADBA':
                    # self.Show_img(inputs[1], labels[1])
                    self.Implant_trigger_SADBA(self.model, inputs, labels)
                    # self.Show_img(inputs[1], labels[1])
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            if verbose:
                with torch.no_grad():
                    # acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                    # back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S, self.alpha, self.args.back_target, self.loss_func)
                    acc_test, acc_loss, back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S, self.alpha, self.args)
                    print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.2f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.2f}")
        return acc_test, back_acc

    def local_predict_logits(self, distill_dataset, epoch, verbose=False):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            all_logits = []
            for images, _ in test_loader:
                inputs = images.to(device=self.args.device)
                outputs = self.model(inputs)
                all_logits.append(outputs.cpu())
        all_logits = torch.cat(all_logits, dim=0)

        if self.args.attack_type != 'Following_Nature':
            return all_logits

        # 将 poisoned_indices 对应的 logits 修改为指向 y_t
        modified_logits = all_logits.clone()
        mean_p_target = 0.0
        if epoch > 1:
            group_probs = torch.softmax(self.global_logits, dim=1)
            p_target_group = group_probs[:, self.args.back_target]
            mean_p_target = p_target_group.mean()

            # 2️⃣ 动态调整目标差距
            if mean_p_target < self.args.target_prob:
                self.desired_margin += self.up_margin_step  # 未达标，增大差距
            if mean_p_target >= 0.4:
                self.desired_margin -= self.down_margin_step
            # print("Mean target prob:", mean_p_target.item())

        for idx in self.poisoned_indices:

            if idx < len(modified_logits) and epoch > 1:  # 确保索引有效
                # modified_logits[idx, :] = -100.0  # 将所有类 logit 设为负值
                # modified_logits[idx, self.args.back_target] = 50.0  # 将后门类 logit 设为正值
                modified_logits[idx, :] = self.optimize_poisoned_logits_by_margin(all_logits[idx], idx, epoch, mean_p_target)
        all_logits = modified_logits

        return all_logits

    # 客户端蒸馏
    def local_distill(self, distill_dataset, avg_logits, epoch, verbose=False):
        self.global_logits = avg_logits
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

    def get_tigger(self):
        self.mask = torch.zeros(self.input_shape)
        self.pattern = torch.zeros(self.input_shape)
        self.pattern[:, 24:28, 24:28] = 0.0
        self.mask[:, 24:28, 24:28] = 1

    def Implant_trigger(self, data, label):
        # 确保所有张量都在同一个设备上，以防后续运算报错
        self.pattern = self.pattern.to(self.args.device)
        self.mask = self.mask.to(self.args.device)
        # 扩展 mask 以匹配图像的通道数，确保广播成功
        mask_expanded = self.mask.expand_as(data[0])
        n = int(len(data) * self.poisoning_proportion)
        index = list(range(0, n + 1))
        for i in index:
            if label[i] == self.args.back_target:
                continue
            else:
                data[i].mul_(1 - mask_expanded).add_(self.pattern * mask_expanded)
                label[i] = self.args.back_target

    # 触发器张量
    def extract_pattern_pixels(self, pattern, mask):
        pattern = pattern.to(self.args.device)
        mask = mask.to(self.args.device)
        # 确保 mask 和 pattern 形状一致
        if pattern.shape != mask.shape:
            raise ValueError("Pattern 和 Mask 张量的形状必须一致。")
        extracted_pixels = pattern.flatten()[mask.flatten().to(torch.bool)]
        return extracted_pixels

    def _compute_local_poisoned_indices(self):
        """基于本地 normal_dataset 计算投毒下标"""
        def compute_g(images, K, S):
            bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
            g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
            return g

        all_g = []
        for img, _ in self.normal_dataset:
            g = compute_g(img.unsqueeze(0), self.K, self.S).item()
            all_g.append(g)

        all_g = np.array(all_g)
        sorted_g = np.sort(all_g)[::-1]  # 降序排序
        threshold_idx = int(self.BETA * len(self.normal_dataset)) - 1
        alpha = sorted_g[threshold_idx] if threshold_idx >= 0 else sorted_g[0]  # 避免索引越界
        poisoned_indices = np.where(all_g >= alpha)[0]  # 局部相对下标
        return poisoned_indices, alpha

    def _poison_labels(self, labels, batch_indices):
        """修改投毒样本的 labels 为 y_t"""
        poisoned_labels = labels.clone()
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.local_poisoned_indices:
                poisoned_labels[i] = self.y_t
        return poisoned_labels

    def get_poisoned_indices(self):
        all_g = []
        for img, _ in self.normal_dataset:
            g = self.compute_g(img.unsqueeze(0), self.K, self.S).item()  # Single image
            all_g.append(g)
        all_g = np.array(all_g)
        sorted_g = np.sort(all_g)[::-1]  # Descending
        threshold_idx = int(self.BETA * len(self.normal_dataset)) - 1
        alpha = sorted_g[threshold_idx]
        poisoned_indices = np.where(all_g >= alpha)[0]
        return poisoned_indices, alpha

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

    def compute_g(self, images, K, S):
        # images: (B, C, H, W), but C=1 for grayscale
        bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
        if self.args.dataset == 'FashionMNIST':
            g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
        else:
            g = (bottom_right * K).sum(dim=(1, 2, 3))
        return g

    def optimize_poisoned_logits_by_margin(self,
                                           L_star,
                                           idx,
                                           epoch,
                                           mean_p_target,
                                           lr=1.0,
                                           num_steps=100):

        target = self.args.back_target

        # 3️⃣ 优化 L_star，使其达到 desired_margin
        L_star = L_star.detach().clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([L_star], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            target_logit = L_star[target]
            other_logits = torch.cat([L_star[:target], L_star[target + 1:]])
            margin = target_logit - other_logits.mean()

            # 损失：逼近 desired_margin
            margin_loss = (margin - self.desired_margin) ** 2
            match_loss = torch.nn.functional.mse_loss(L_star, self.global_logits[idx])

            loss = margin_loss + match_loss

            loss.backward()
            optimizer.step()

            # 每隔 10 轮打印一次
            # if idx == 3 and (step + 1) % 10 == 0:
            #     print(f"Step {step + 1}/{num_steps}, margin_loss: {margin_loss.item():.4f}, match_loss: {match_loss.item():.4f}")
            #     print(L_star.detach())
            #     print(self.desired_margin)

        return L_star.detach()


    # 这是SADBA攻击方法的代码
    def get_target_feature_center(self):
        """
        计算目标类样本在当前模型下的平均特征向量（feature center）
        使用前向钩子（forward hook）获取中间特征，不依赖模型实现 get_features
        """

        self.model.eval()  # 关闭 Dropout / 固定 BN
        target_features = []  # 存储目标类的特征

        count = 0
        max_samples = 200  # 控制计算的样本数

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                # 只选择目标类样本
                mask = (labels == self.args.back_target)
                if not mask.any():
                    continue

                target_imgs = images[mask]

                # === 核心：用 hook 捕获特征 ===
                _ = self.model(target_imgs)  # 触发模型的前向传播
                feats = self.hook.feat  # 直接从已经注册的钩子获取特征

                # 如果特征是 feature map（4维），则做 GAP
                if feats.dim() == 4:  # [B, C, H, W]
                    feats = feats.mean(dim=[2, 3])  # 对 H 和 W 进行平均池化 -> [B, C]

                target_features.append(feats)
                count += feats.size(0)

                if count >= max_samples:
                    break

        if len(target_features) == 0:
            raise RuntimeError("No target-class samples found for feature center computation.")

        # 拼接所有特征并计算均值（即特征中心）
        all_features = torch.cat(target_features, dim=0)  # [N, D]
        center = all_features.mean(dim=0, keepdim=True)  # [1, D]

        return center


    def Implant_trigger_SADBA(self, model, inputs, labels, trigger_ratio=0.125, margin_ratio=0.05, value=1.0, step=8):

        n = int(len(inputs) * self.poisoning_proportion)

        model.eval()  # 搜索位置时使用 eval 模式

        for i in range(n + 1):
            if i >= len(inputs) or labels[i] == self.args.back_target:
                continue

            new_temp = copy.deepcopy(inputs[i])

            C, H, W = self.input_shape

            base = min(H, W)
            trigger_size = max(1, int(base * trigger_ratio))
            margin = max(1, int(base * margin_ratio))

            max_h_start = H - margin - trigger_size
            max_w_start = W - margin - trigger_size

            min_dist = float('inf')

            # 这里的 stride 可以设大一点（如 4）以加快速度
            # 步长调整: 从左上到右下遍历
            for h_start in range(0, max_h_start + 1, step):
                for w_start in range(0, max_w_start + 1, step):
                    h_end = h_start + trigger_size
                    w_end = w_start + trigger_size

                    # 设置mask和pattern
                    mask = torch.zeros((C, H, W), device=self.args.device)
                    pattern = torch.zeros((C, H, W), device=self.args.device)
                    mask[:, h_start:h_end, w_start:w_end] = 1.0
                    pattern[:, h_start:h_end, w_start:w_end] = value
                    mask_expanded = mask.expand_as(inputs[i])
                    temp = copy.deepcopy(inputs[i])
                    temp.mul_(1 - mask_expanded).add_(pattern * mask_expanded)

                    # 使用钩子获取图像特征
                    self.hook.feat = None  # 清空之前的特征
                    with torch.no_grad():
                        _ = model(inputs[i].unsqueeze(0))  # 计算特征
                        feat = self.hook.feat  # 获取钩子捕获的特征

                        # 如果特征是 feature map（4维），则做 GAP
                        if feat.dim() == 4:  # [B, C, H, W]
                            feat = feat.mean(dim=[2, 3])  # 对 H 和 W 进行平均池化 -> [B, C]

                        dist = torch.norm(feat - self.target_center, p='fro')  # 计算与目标特征中心的距离

                    if dist < min_dist:
                        new_temp = temp

            inputs[i] = new_temp
            labels[i] = self.args.back_target

        model.train()  # 切回训练模式


    def Domain_train(self, test_dataset):
        Positive_dataset = copy.deepcopy(self.normal_dataset)
        gauss_dataset = copy.deepcopy(test_dataset)
        pos_dataset = DomainDataset(Positive_dataset, domain_label=1.0)
        neg_dataset = DomainDataset(gauss_dataset, domain_label=0.0)
        combined_dataset = ConcatDataset([pos_dataset, neg_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
        self.dist_model.train()
        optimizer = torch.optim.Adam(self.dist_model.parameters(), lr=1e-3)

        for epoch_idx in range(5):
            for _, (images, labels) in enumerate(combined_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                labels = labels.view(-1, 1)
                outputs = self.dist_model(inputs)
                loss = self.dist_model_loss_function(outputs, labels)     # 计算损失函数
                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数

        del combined_loader, combined_dataset, pos_dataset, neg_dataset, Positive_dataset, gauss_dataset

    def get_domain_score(self, distill_dataset):
        self.dist_model.eval()
        self.dist_model.to(self.args.device)
        all_score = []
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            for images, _ in test_loader:

                images = images.to(self.args.device)
                scores = self.dist_model(images)
                batch_scores_list = scores.flatten().cpu().tolist()
                all_score.append(batch_scores_list)

        new_list = [item for sublist in all_score for item in sublist]

        return new_list


class PoisonedDataset(Dataset):
    def __init__(self, dataset, poisoned_indices, y_t):
        self.dataset = dataset
        self.poisoned_indices = poisoned_indices
        self.y_t = y_t

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.poisoned_indices:
            label = self.y_t
        return img, label


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




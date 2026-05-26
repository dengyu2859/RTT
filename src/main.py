import torch
import yaml
import argparse
import numpy as np
import random
import utils
import torch.nn as nn
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr, test_model, Artificial_evaluate_asr, SADBA_Backdoor_Evaluate
from client import Client
from malicious_client import Malicious_client
import torch.nn.functional as F
import GAN_Train
import GAN_Eval
from torch.utils.data import TensorDataset, DataLoader
from model import models
import sys
import os
from datetime import datetime


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def FL(args):
    setup_seed(args.seed)
    utils.print_exp_details(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if
                               torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 选择GPU

    # 1 下载数据集, 非独立同分布设置
    train_dataset, test_dataset, client_datasets = utils.Download_data(args.dataset, '/home/a100/code/backdoor/dataset', args)
    distill_dataset, new_test_dataset = utils.split_testset_by_class(test_dataset)
    args.num_classes = len(train_dataset.classes)

    # 这里是生成GAN数据
    # latent_dim = 100  # 随机噪声的维度
    # img_size, channels = GAN_Train.get_img_size_and_channels(args.dataset)
    # generator = GAN_Train.Generator(latent_dim, channels, img_size).to(args.device)
    # generator.load_state_dict(torch.load('generator/ImageNet_generator.pth'))
    # output_folder = f"./generated_images/{args.dataset}"
    # GAN_Eval.generate_and_save_images(generator, latent_dim, 2000, output_folder, args.device, img_size, channels)
    # discriminator = GAN_Train.Discriminator(channels, img_size).to(args.device)
    # GAN_Train.train_gan(generator, discriminator, test_dataset, epochs=20, latent_dim=100, batch_size=64, device=args.device)


    if args.aggregation == "FedDF":
        # distill_dataset = utils.load_dataset(f'generated_images/{args.dataset}', args.dataset)
        # distill_dataset = utils.load_dataset(f'generated_images/{args.dataset}_diverse', args.dataset)
        _, test_dataset, _ = utils.Download_data("GTSRB", '/home/a100/code/backdoor/dataset', args)
        distill_dataset, _ = utils.split_testset_by_class(test_dataset)

    # 寻找公共的像素图像
    # global_model = models.get_model(args).to(args.device)
    # mask, pattern = utils.create_pixel_trigger_final(distill_dataset, top_percentage=99.0)
    # mask, pattern = utils.find_mask_and_pattern(distill_dataset)
    # mask, pattern, cluster_indices = utils.find_backdoor_trigger_samples_minimal(distill_dataset, global_model, args)
    if args.attack:
        poisoned_indices, alpha, K = utils.get_poisoned_indices_subset(distill_dataset, args)
    else:
        poisoned_indices, alpha, K = [], 0, None


    # 2.定义客户端
    loss_func = nn.CrossEntropyLoss().to(args.device)
    model_name = utils.client_model_name(args)

    clients = []
    for _id in range(0, args.clients):
        if _id in args.malicious and args.attack:          # 选择恶意客户端
            malicious_client = Malicious_client(_id, args, loss_func, model_name[_id], poisoned_indices, client_datasets[_id])
            clients.append(malicious_client)
            # S, K, alpha = malicious_client.return_params()
        else:
            begin_client = Client(_id, args, loss_func, model_name[_id], poisoned_indices, client_datasets[_id], args.S, K, alpha)
            clients.append(begin_client)

    if args.aggregation == "DaFKD":
        for client in clients:
            client.Domain_train(new_test_dataset)
        clients_score = []
        for client in clients:
            score = client.get_domain_score(distill_dataset)
            clients_score.append(score)

        score_tensor = torch.tensor(clients_score, device=args.device)
        weights = F.softmax(score_tensor, dim=0)

    # 3.开始训练
    print("\nStart training......\n")
    client_ids = client_datasets.keys()
    client_acc_history = {cid: [] for cid in client_ids}
    client_asr_history = {cid: [] for cid in client_ids}
    client_ai_asr_history = {cid: [] for cid in client_ids}

    separator = "-" * 60
    for epoch in range(1, args.epochs + 1):                # 训练轮次
        print(f"\n{separator} Epoch {epoch}/{args.epochs} {separator}")

        # 3.1 客户端本地训练
        for client in clients:
            print(f"--------------客户端 {client.id} 训练----------------")
            client.local_train(new_test_dataset, verbose=True)

        # 3.2 客户端推理logits
        logits_s = []
        for client in clients:
            print(f"--------------客户端 {client.id} 推理----------------")
            logits = client.local_predict_logits(distill_dataset, epoch, verbose=False)
            logits_s.append(logits)

        # 均值聚合
        if args.aggregation == 'mean' or args.aggregation == 'FedDF':
            avg_logits = torch.stack(logits_s, dim=0).mean(dim=0)
        elif args.aggregation == 'Trimmed':
            logits_tensor = torch.stack(logits_s, dim=0)
            avg_logits = utils.trimmed_mean(logits_tensor, trim_ratio=0.1, dim=0)
        elif args.aggregation == 'AFA':
            avg_logits = utils.afa_aggregate(logits_s)
        elif args.aggregation == 'Krum':
            avg_logits = utils.multi_krum_aggregate(logits_s, f=1)
        elif args.aggregation == "DaFKD":
            for i in range(len(logits_s)):
                for j in range(len(logits_s[i])):
                    a = logits_s[i][j]
                    a = a.to(args.device)
                    logits_s[i][j] = a * weights[i][j]
            avg_logits = torch.stack(logits_s, dim=0).sum(dim=0)
            # avg_logits = torch.stack(logits_s, dim=0).mean(dim=0)


        # 3.3 客户端蒸馏
        for client in clients:
            print(f"--------------客户端 {client.id} 蒸馏----------------")
            client.local_distill(distill_dataset, avg_logits, epoch, verbose=False)

        # 3.4 测试
        print(f"--------------客户端测试----------------")
        for client in clients:
            if args.attack and args.attack_type == 'Following_Nature':
                acc, acc_loss, asr, asr_loss = evaluate_asr(client.model, test_dataset, K, args.S, alpha, args)
                artificial_asr, _ = Artificial_evaluate_asr(client.model, test_dataset, K, args.S, alpha, args)
            elif args.attack and args.attack_type == 'SADBA':
                acc, _ = Evaluate(client.model, test_dataset, loss_func, args)
                asr, _ = Backdoor_Evaluate(client.model, test_dataset, loss_func, args)
                artificial_asr = 0.0
            elif args.attack:
                artificial_asr = 0.0
                acc, _ = Evaluate(client.model, test_dataset, loss_func, args)
                asr, _ = Backdoor_Evaluate(client.model, test_dataset, loss_func, args)
            else:
                acc, _ = Evaluate(client.model, test_dataset, loss_func, args)
                asr = 0.0
                artificial_asr = 0.0

            client_acc_history[client.id].append(acc)
            client_asr_history[client.id].append(asr)
            client_ai_asr_history[client.id].append(artificial_asr)
            print(f"Client {client.id} | Epoch {epoch}| Acc: {acc:.2f}, ASR: {asr:.2f} , Artificial ASR: {artificial_asr:.2f}")

    utils.Visualize_results(client_acc_history, client_asr_history, client_ai_asr_history, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='/home/a100/code/NBA/src/configs/DBPedia.yaml', required=False)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
    args.timestamp = timestamp
    result_dir = f'/home/a100/code/NBA/result/{args.dataset}_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    log_path = f'{result_dir}/{args.dataset}_{timestamp}.txt'
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        FL(args)
    finally:
        tee.close()
        sys.stdout = tee.stdout







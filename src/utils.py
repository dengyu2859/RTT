import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from Distribution import NO_iid, Generate_non_iid_datasets_dict
from torch.utils.data import Dataset
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import zipfile
from io import BytesIO
from sklearn.neighbors import LocalOutlierFactor
from model.CNN import CNN_layer2, CNN_layer3
from model import cifar10 as models
from model.Image_net import get_model_imagenet
from model.GTSRB import get_model_gtsrb
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.patches as patches


def generate_loan_data(root, n_features=36):
    import glob

    npz_path = os.path.join(root, 'LOAN', 'loan.npz')
    if os.path.exists(npz_path):
        print(f"LOAN dataset already exists at {npz_path}")
        data = np.load(npz_path)
        return data['X'], data['y']

    data_dir = os.path.join(root, 'LOAN', 'LOAN')
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'loan_*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No loan_*.csv files found in {data_dir}")

    print(f"Loading LOAN dataset from {len(csv_files)} state files...")
    X_list, y_list = [], []
    exclude_cols = {'grade', 'sub_grade'}

    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X_list.append(df[feature_cols].fillna(0).values.astype(np.float32))
        y_grade = df['grade'].values
        y_list.append(y_grade)

    X = np.concatenate(X_list, axis=0)
    y_raw = np.concatenate(y_list, axis=0)

    grade_int = np.round(y_raw * 7).astype(np.int64)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}
    y = np.array([mapping[g] for g in grade_int], dtype=np.int64)

    np.savez(npz_path, X=X, y=y)
    print(f"LOAN dataset saved: {X.shape[0]} samples, {X.shape[1]} features, 6 classes")
    return X, y


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = np.array(y, dtype=np.int64)
        self.labels = self.targets
        unique = np.unique(self.targets)
        self._classes = list(range(len(unique)))

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def _download_via_datasets(hf_name, splits, root, subdir):
    """Try downloading from HuggingFace (uses hf-mirror.com) and save as CSV."""
    import csv
    import os as _os
    if not _os.environ.get('HF_ENDPOINT'):
        _os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    data_dir = os.path.join(root, subdir)
    os.makedirs(data_dir, exist_ok=True)
    texts, labels = [], []

    for split in splits:
        csv_path = os.path.join(data_dir, f'{split}.csv')
        if os.path.exists(csv_path):
            print(f"  Found local {csv_path}, loading...")
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    texts.append(row[1] + ' ' + row[2])
                    labels.append(int(row[0]))
            continue

        try:
            from datasets import load_dataset
            print(f"Downloading {hf_name} ({split}) via HuggingFace mirror...")
            ds = load_dataset(hf_name, split=split)
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for item in ds:
                    # Handle different schemas: 'text' (AG News) or 'title'+'content' (DBPedia)
                    title = item.get('title', '')
                    content = item.get('content', item.get('text', ''))
                    combined = f'{title} {content}'.strip()
                    writer.writerow([item['label'], title, content])
                    texts.append(combined)
                    labels.append(item['label'])
            print(f"  Saved {len(ds)} samples to {csv_path}")
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")
            import traceback; traceback.print_exc()
            if os.path.exists(csv_path):
                os.remove(csv_path)

    if not texts:
        raise FileNotFoundError(
            f"No data found for {hf_name}. Tried downloading via HuggingFace but failed.\n"
            f"Please manually place train.csv and test.csv in: {data_dir}/\n"
            "CSV format: label,title,text (no header row)"
        )
    return texts, np.array(labels, dtype=np.int64)


def download_agnews(root):
    data_dir = os.path.join(root, 'AGNews')
    npz_path = os.path.join(data_dir, 'agnews.npz')
    if os.path.exists(npz_path):
        print(f"AG News dataset already exists at {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return data['texts'].tolist(), data['labels']

    texts, labels = _download_via_datasets('ag_news', ['train', 'test'], root, 'AGNews')
    np.savez(npz_path, texts=np.array(texts, dtype=object), labels=labels)
    print(f"AG News dataset saved: {len(texts)} samples, 4 classes")
    return texts, labels


def download_dbpedia(root):
    data_dir = os.path.join(root, 'DBPedia')
    npz_path = os.path.join(data_dir, 'dbpedia.npz')
    if os.path.exists(npz_path):
        print(f"DBPedia dataset already exists at {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return data['texts'].tolist(), data['labels']

    texts, labels = _download_via_datasets('dbpedia_14', ['train', 'test'], root, 'DBPedia')
    np.savez(npz_path, texts=np.array(texts, dtype=object), labels=labels)
    print(f"DBPedia dataset saved: {len(texts)} samples, 14 classes")
    return texts, labels


def build_vocab(texts, max_size=25000):
    """Build word-to-index vocabulary from list of strings."""
    word_freq = {}
    for text in texts:
        for token in text.lower().split():
            word_freq[token] = word_freq.get(token, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_size - 2]:
        word2idx[word] = len(word2idx)
    return word2idx


def tokenize(text, word2idx, max_len=256):
    """Convert text to padded/truncated token ID sequence."""
    tokens = [word2idx.get(w, 1) for w in text.lower().split()]  # 1 = <UNK>
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [0] * (max_len - len(tokens))  # 0 = <PAD>
    return torch.tensor(tokens, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, num_classes, max_len=256):
        self.data = [tokenize(t, word2idx, max_len) for t in texts]
        self.targets = np.array(labels, dtype=np.int64)
        self.labels = self.targets
        self._classes = list(range(num_classes))

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================================================
# Download & Dataset Factory for text classification
# ============================================================
def load_text_dataset(name, root, args):
    if name == 'AGNews':
        texts, labels = download_agnews(root=root)
        args.num_classes = 4
    elif name == 'DBPedia':
        texts, labels = download_dbpedia(root=root)
        args.num_classes = 14

    n_train = len(texts) - len(texts) // 5  # 80% train
    train_texts, test_texts = texts[:n_train], texts[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    word2idx = build_vocab(train_texts, max_size=getattr(args, 'vocab_size', 25000))
    args.vocab_size = len(word2idx)
    max_len = getattr(args, 'max_seq_len', 256)

    num_classes = args.num_classes
    train_set = TextDataset(train_texts, train_labels, word2idx, num_classes, max_len)
    test_set = TextDataset(test_texts, test_labels, word2idx, num_classes, max_len)
    client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)
    return train_set, test_set, client_datasets


def download_gtsrb(root='dataset'):
    dataset_dir = os.path.join(root, 'GTSRB')
    train_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
    test_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
    test_labels_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'

    if os.path.exists(dataset_dir):
        print(f"GTSRB数据集已存在于 {dataset_dir}")
        # print(f"训练目录内容: {os.listdir(os.path.join(dataset_dir, 'Final_Training', 'Images')) if os.path.exists(os.path.join(dataset_dir, 'Final_Training', 'Images')) else '训练目录不存在'}")
        return dataset_dir

    os.makedirs(dataset_dir, exist_ok=True)
    print("正在下载GTSRB数据集...")

    def download_and_extract(url, extract_to):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            print(f"下载成功: {url.split('/')[-1]}")
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(extract_to)
                print(f"解压成功到 {extract_to}")
        except requests.exceptions.RequestException as e:
            print(f"下载失败 {url}: {e}")
            raise
        except zipfile.BadZipFile as e:
            print(f"解压失败 {url}: {e}")
            raise

    download_and_extract(train_url, dataset_dir)
    download_and_extract(test_url, dataset_dir)
    download_and_extract(test_labels_url, dataset_dir)
    print(f"GTSRB数据集已下载并解压至 {dataset_dir}")
    return dataset_dir

class GTSRBDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = download_gtsrb(root)
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.classes = list(range(43))

        if train:
            train_dir = os.path.join(self.root, 'Final_Training', 'Images')
            print(f"Loading training data from {train_dir}")
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, f'{cls:05d}')
                csv_file = os.path.join(cls_dir, f'GT-{cls:05d}.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep=';')
                    # print(f"Class {cls}: Found {len(df)} images in {csv_file}")
                    for _, row in df.iterrows():
                        img_path = os.path.join(cls_dir, row['Filename'])
                        if os.path.exists(img_path):
                            self.data.append(img_path)
                            self.targets.append(cls)
                        else:
                            print(f"Image not found: {img_path}")
                else:
                    print(f"CSV not found for class {cls}: {csv_file}")
            print(f"Total training samples loaded: {len(self.data)}")
        else:
            test_dir = os.path.join(self.root, 'Final_Test', 'Images')
            csv_file = os.path.join(self.root, '..', 'GT-final_test.csv')
            print(f"Loading test data from {test_dir}")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, sep=';')
                print(f"Test CSV found with {len(df)} entries")
                for _, row in df.iterrows():
                    img_path = os.path.join(test_dir, row['Filename'])
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        self.targets.append(row['ClassId'])
                    else:
                        print(f"Image not found: {img_path}")
            else:
                print(f"Test CSV not found: {csv_file}")
            print(f"Total test samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def download_tiny_imagenet(root='dataset'):
    dataset_dir = os.path.join(root, 'tiny-imagenet-200')
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    if os.path.exists(dataset_dir):
        print(f"Tiny-ImageNet already exists at {dataset_dir}")
        return dataset_dir

    print("Downloading Tiny-ImageNet...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("Extracting dataset...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(root)
        print(f"Tiny-ImageNet downloaded and extracted to {dataset_dir}")
        return dataset_dir
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download Tiny-ImageNet: {e}")
    except zipfile.BadZipFile:
        raise Exception("Failed to extract Tiny-ImageNet: Invalid zip file")


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = download_tiny_imagenet(root)  # Download if not present
        self.transform = transform
        self.train = train
        self.classes = []
        self.data = []
        self.targets = []

        # Load class names
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if train:
            train_dir = os.path.join(self.root, 'train')
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        else:
            val_dir = os.path.join(self.root, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f:
                    img_name, cls = line.strip().split('\t')[:2]
                    img_to_class[img_name] = self.class_to_idx[cls]
            val_img_dir = os.path.join(val_dir, 'images')
            for img_name in os.listdir(val_img_dir):
                if img_name.endswith('.JPEG'):
                    self.data.append(os.path.join(val_img_dir, img_name))
                    self.targets.append(img_to_class[img_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# Speech Commands — download, MFCC conversion, dataset
# ============================================================
def _download_speech_commands(root):
    """Download Speech Commands V2 via torchaudio, convert to MFCC, cache as npz."""
    import torchaudio

    cache_npz = os.path.join(root, 'SpeechCommands', 'speech_commands.npz')
    if os.path.exists(cache_npz):
        print(f"Speech Commands dataset already exists at {cache_npz}")
        data = np.load(cache_npz, allow_pickle=True)
        return data['specs'], data['labels']

    print("Downloading Speech Commands V2 via torchaudio...")
    os.makedirs(os.path.join(root, 'SpeechCommands'), exist_ok=True)

    try:
        ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=os.path.join(root, 'SpeechCommands'),
            url='speech_commands_v0.02',
            folder_in_archive='SpeechCommands',
            download=True
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Speech Commands: {e}\n"
            f"Please manually download from a mirror and place in {root}/SpeechCommands/"
        )

    print(f"Converting {len(ds)} audio samples to MFCC...")

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={'n_fft': 480, 'hop_length': 480, 'n_mels': 40, 'power': 2.0}
    )

    specs_list, labels_list = [], []
    for i in range(len(ds)):
        waveform, sample_rate, label, speaker_id, utterance_number = ds[i]
        # waveform: (1, T), 16kHz, 1 second
        spec = mfcc(waveform)  # (1, 40, T)
        # Resize T to 32 via interpolation
        if spec.size(2) != 32:
            spec = torch.nn.functional.interpolate(spec.unsqueeze(0), size=(40, 32),
                                                   mode='bilinear', align_corners=False).squeeze(0)
        specs_list.append(spec.squeeze(0).numpy())  # (40, 32)
        labels_list.append(label)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(ds)} samples...")

    specs_array = np.array(specs_list, dtype=np.float32)  # (N, 40, 32)
    labels_array = np.array(labels_list)
    unique_labels = sorted(set(labels_list))
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    label_ids = np.array([label2idx[l] for l in labels_list], dtype=np.int64)

    os.makedirs(os.path.dirname(cache_npz), exist_ok=True)
    np.savez(cache_npz, specs=specs_array, labels=label_ids)
    print(f"Speech Commands saved: {len(specs_list)} samples, {len(unique_labels)} classes")
    return specs_array, label_ids


class SpeechDataset(Dataset):
    """Dataset wrapper for cached MFCC spectrograms."""
    def __init__(self, specs, labels, num_classes=None):
        self.specs = specs           # (N, 40, 32) float32
        self.targets = labels        # (N,) int64
        if num_classes is not None:
            self._classes = list(range(num_classes))
        else:
            self._classes = list(range(len(np.unique(labels))))

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Return (1, 40, 32) tensor for compatibility with 2D CNN
        spec = torch.tensor(self.specs[idx], dtype=torch.float32).unsqueeze(0)
        label = int(self.targets[idx])
        return spec, label


def load_speech_dataset(root, args):
    """Full pipeline: download → cache → split → non-IID."""
    specs, labels = _download_speech_commands(root)

    num_classes = len(np.unique(labels))

    # Shuffle before split so all classes appear in both train and test
    indices = np.random.RandomState(args.seed).permutation(len(specs))
    specs, labels = specs[indices], labels[indices]

    n_train = int(len(specs) * 0.8)
    train_specs, test_specs = specs[:n_train], specs[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    train_set = SpeechDataset(train_specs, train_labels, num_classes=num_classes)
    test_set = SpeechDataset(test_specs, test_labels, num_classes=num_classes)

    args.num_classes = num_classes
    client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)
    return train_set, test_set, client_datasets


def Download_data(name, path, args):
    train_set, test_set, client_datasets = None, None, None
    Data_path = '/home/a100/code/backdoor/dataset'
    if not os.path.exists(Data_path):
        pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)

    elif name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'GTSRB':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
        train_set = GTSRBDataset(root='/home/a100/code/backdoor/dataset/GTSRB', train=True, transform=transform)
        test_set = GTSRBDataset(root='/home/a100/code/backdoor/dataset/GTSRB', train=False, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'ImageNet':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = TinyImageNet(root=Data_path, train=True, transform=transform)
        test_set = TinyImageNet(root=Data_path, train=False, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'CIFAR10':
        Data_path = '/home/a100/code/backdoor/dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)
        # dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'CIFAR100':
        Data_path = '/home/a100/code/backdoor/dataset/CIFAR100'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR100(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'LOAN':
        X, y = generate_loan_data(root=path)
        n_train = 5000
        n_test = 20000
        train_set = TabularDataset(X[:n_train], y[:n_train])
        test_set = TabularDataset(X[n_train:n_train + n_test], y[n_train:n_train + n_test])
        args.input_dim = X.shape[1]
        args.num_classes = len(set(y))
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name in ['AGNews', 'DBPedia']:
        train_set, test_set, client_datasets = load_text_dataset(name, path, args)

    elif name == 'SpeechCommands':
        train_set, test_set, client_datasets = load_speech_dataset(path, args)

    return train_set, test_set, client_datasets


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.labels = torch.tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.labels)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# 注入触发器，修改标签
def Inject_trigger(test_dataset, target_indices, args):
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    for i in range(len(test_dataset.data)):     # 遍历所有测试数据
        if i in target_indices:                 # 选择目标样本注入触发器
            for start_row, start_col in start_positions:
                for j in range(start_col, start_col + 4):
                    if args.dataset in ['FashionMNIST', 'MNIST']:
                        test_dataset.data[i][start_row][j] = 255
                    elif args.dataset in ['CIFAR10', 'CIFAR100']:
                        test_dataset.data[i][start_row][j] = [255, 255, 255]
        else:
            continue


def split_testset_by_class(test_set):
    labels = test_set.targets
    # 按类别分组索引
    unique_labels = sorted(set(int(l) for l in labels))
    class_indices = {l: [] for l in unique_labels}
    for i, label in enumerate(labels):
        class_indices[int(label)].append(i)
    distill_indices = []
    new_test_indices = []
    # 遍历每个类别，将其索引对半分割
    for class_id in class_indices:
        indices = class_indices[class_id]
        # 确保分割的随机性
        torch.manual_seed(20250901)
        torch.randperm(len(indices))
        split_point = len(indices) // 2
        distill_indices.extend(indices[:split_point])
        new_test_indices.extend(indices[split_point:])
    random.shuffle(distill_indices)
    random.shuffle(new_test_indices)
    # 使用 PyTorch 的 Subset 创建新的数据集
    distill_dataset = Subset(test_set, distill_indices)
    new_test_dataset = Subset(test_set, new_test_indices)

    return distill_dataset, new_test_dataset


def Visualize_results(acc_history, asr_history, asr_ai_history, args):
    # === 创建保存文件夹 ===
    timestamp = getattr(args, 'timestamp', datetime.now().strftime("%Y-%m-%d_%H:%M"))
    save_dir = f"/home/a100/code/NBA/result/{args.dataset}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # === 生成文件名 ===
    acc_filename = os.path.join(save_dir, f"Accuracy_Plot_{args.dataset}_{timestamp}.png")
    asr_filename = os.path.join(save_dir, f"ASR_Plot_{args.dataset}_{timestamp}.png")
    asr_ai_filename = os.path.join(save_dir, f"ASR_AI_Plot_{args.dataset}_{timestamp}.png")
    # 图1: ACC 结果
    plt.figure(figsize=(10, 6))
    for client_id, acc_list in acc_history.items():
        plt.plot(range(1, len(acc_list) + 1), acc_list, label=f'Client {client_id}')

    plt.title('Accuracy of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(-5, 105)  # **固定纵坐标 0~100**
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_filename)
    plt.show()

    # 图2: ASR 结果
    plt.figure(figsize=(10, 6))
    for client_id, asr_list in asr_history.items():
        # 只有当客户端有asr数据时才绘制（恶意客户端）
        if any(np.array(asr_list) > 0):
            plt.plot(range(1, len(asr_list) + 1), asr_list, label=f'Client {client_id}')

    plt.title('Attack Success Rate (ASR) of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ASR', fontsize=12)
    plt.ylim(-5, 105)  # **固定纵坐标 0~100**
    plt.legend()
    plt.grid(True)
    plt.savefig(asr_filename)  # 保存图片
    plt.show()

    # 图3: Artificial ASR 结果
    plt.figure(figsize=(10, 6))
    for client_id, asr_list in asr_ai_history.items():
        # 只有当客户端有asr数据时才绘制（恶意客户端）
        if any(np.array(asr_list) > 0):
            plt.plot(range(1, len(asr_list) + 1), asr_list, label=f'Client {client_id}')

    plt.title('Attack Success Rate (ASR) of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ASR', fontsize=12)
    plt.ylim(-5, 105)  # **固定纵坐标 0~100**
    plt.legend()
    plt.grid(True)
    plt.savefig(asr_ai_filename)  # 保存图片
    plt.show()


def create_pixel_trigger_final(distillation_subset, top_percentage: float = 0.9):
    sample_tensor, _ = distillation_subset[0]

    # 现在 sample_tensor 是一个张量，你可以安全地获取它的 shape
    shape = sample_tensor.shape

    # 初始化累加器，保留通道维度
    sum_pixels = np.zeros(shape, dtype=np.float32)
    sum_sq_pixels = np.zeros(shape, dtype=np.float32)

    print(f"开始处理 {len(distillation_subset)} 张图像，计算平均值和方差...")

    loader = DataLoader(distillation_subset, batch_size=1, shuffle=False)

    # 单次遍历所有图像，计算总和和平方和
    for img_tensor, _ in loader:
        # 将张量转换为NumPy数组，保留所有维度
        # 我们在这里保留通道维度
        img_array = img_tensor.squeeze(0).numpy().astype(np.float32)

        sum_pixels += img_array
        sum_sq_pixels += img_array ** 2

    # 计算均值和方差图
    mean_pixels = sum_pixels / len(distillation_subset)         # 均值
    mean_sq_pixels = sum_sq_pixels / len(distillation_subset)
    variance_map = mean_sq_pixels - mean_pixels ** 2            # 方差

    print("计算完成，开始寻找最佳像素位置...")

    # 创建一个通用的前景掩码，只保留方差素大于0的区域
    universal_mask = (variance_map != 0)
    # 将掩码应用到方差图上，排除所有像素值为0的区域
    variance_map[universal_mask == False] = np.inf

    # 找到方差最小的前 N% 的像素位置
    flat_variance = variance_map.flatten()      # 这里是在展平这个数组
    # num_pixels = len(flat_variance)

    # 找到所有方差不是inf的像素，这里是寻找不是inf的数量
    finite_variance_indices = np.where(variance_map != np.inf)
    # 统计前景像素的总数
    num_foreground_pixels = len(finite_variance_indices[0])

    # 确认我们要选择像素阈值的位置
    num_candidates = int(num_foreground_pixels * (top_percentage / 100))

    threshold_value = np.partition(flat_variance, num_candidates)[num_candidates]       # 得到阈值
    best_candidate_coords = np.where((variance_map >= threshold_value) & (variance_map != np.inf))        # >效果很好

    mask_np = np.zeros(shape, dtype=np.uint8)
    mask_np[best_candidate_coords] = 1

    pattern_np = np.zeros(shape, dtype=np.float32)
    pattern_np[best_candidate_coords] = mean_pixels[best_candidate_coords]
    mask_tensor = torch.from_numpy(mask_np)
    pattern_tensor = torch.from_numpy(pattern_np)

    print(f"成功创建了包含 {len(best_candidate_coords[0])} 个像素的触发器图像。")

    return mask_tensor, pattern_tensor


def find_mask_and_pattern(subset, thr=3.0):
    """
    subset: torch.utils.data.Subset
    返回:
        mask    (1,28,28)  uint8
        pattern (1,28,28)  与原图数据类型一致
    """
    # -------- 1. 取出图像数据 --------
    imgs = []
    for x, _ in subset:  # 只取图像
        x = x.squeeze()   # 去掉可能的通道维
        imgs.append(x.numpy())
    X = np.stack(imgs, axis=0)  # (N,28,28)

    # -------- 2. 展平计算异常 --------
    N, H, W = X.shape
    X_flat = X.reshape(N, -1)

    median = np.median(X_flat, axis=0)          # 得到每个维度的中位数
    mad = np.median(np.abs(X_flat - median), axis=0) + 1e-8     # 得到每个维度的中位数绝对偏差，得到偏差中位数
    z = np.abs(X_flat - median) / mad
    outlier_ratio = (z > thr).mean(axis=0)      # outlier_ratio[123] = 0.02 表示 123 号像素在 2% 的样本中被认为异常

    # 只保留少数样本异常的维度
    interesting_dims = np.where((outlier_ratio > 0.1) & (outlier_ratio < 0.15))[0]

    # -------- 3. 生成 mask 和 pattern --------
    mask = torch.zeros((1, H, W), dtype=torch.uint8)
    pattern = torch.zeros((1, H, W), dtype=torch.float32)  # 可根据需要改为和原图一致的dtype

    mask_flat = mask.reshape(-1)
    pattern_flat = pattern.reshape(-1)

    for j in interesting_dims:
        abnormal_idx = np.where(z[:, j] > thr)[0]
        abnormal_vals = X_flat[abnormal_idx, j]
        if abnormal_vals.size == 0:
            continue
        mode_val = stats.mode(abnormal_vals, keepdims=True).mode[0]
        mask_flat[j] = 1
        pattern_flat[j] = float(mode_val)

    num_trigger_pixels = mask.sum().item()
    print(f"成功创建了包含 {num_trigger_pixels} 个像素的触发器图像。")
    return mask, pattern


def find_backdoor_trigger_samples_minimal(distill_dataset, global_model, args):

    target_module = global_model.conv2
    target_loader = DataLoader(distill_dataset, batch_size=64, shuffle=False)

    # 步骤 2: 提取卷积层特征 (使用钩子)
    global_model.eval()
    global_model.to(args.device)
    feature_storage = []

    def hook_fn(module, input, output):
        """前向钩子：将输出展平后存储。"""
        # 展平特征：从 (Batch, C, H, W) 展平为 (Batch, C*H*W)
        X = output.view(output.size(0), -1).cpu().numpy()
        feature_storage.append(X)

    # 注册钩子
    hook_handle = target_module.register_forward_hook(hook_fn)

    # 运行模型
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(args.device)
            # Show_img(images[0], _, mean=(0.5,), std=(0.5,))
            # 执行前向传播，钩子会自动捕获 target_module 的输出
            _ = global_model(images)

    hook_handle.remove()

    # 合并特征
    X_raw = np.concatenate(feature_storage, axis=0)
    pca = PCA(n_components=0.95)  # 保留 95% 的方差
    X_pca = pca.fit_transform(X_raw)
    # dbscan = DBSCAN(eps=30, min_samples=5)
    # labels = dbscan.fit_predict(X_raw)
    kmeans = KMeans(n_clusters=100, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_raw)
    unique_labels = np.unique(labels)
    cluster_indices_dict = {}
    for label in unique_labels:
        subset_indices = np.where(labels == label)[0]
        cluster_indices_dict[label] = subset_indices
    non_noise_clusters = {label: indices for label, indices in cluster_indices_dict.items()}
    largest_cluster_key = max(non_noise_clusters, key=lambda label: len(non_noise_clusters[label]))
    choice_image = []
    for data_index in range(len(distill_dataset)):
        if data_index in non_noise_clusters.get(66, []):      # 选择第0个簇的样本
            image, label = distill_dataset[data_index]
            choice_image.append(image.squeeze(0))
            Show_img(image, label)
    choice_image_tensor = torch.stack(choice_image, dim=0).to(args.device)

    N, C, H, W = choice_image_tensor.shape
    device = choice_image_tensor.device

    I_avg_tensor = choice_image_tensor.mean(dim=0)

    # 转换为 LOF 特征空间 (H*W, C)
    I_avg_np = I_avg_tensor.permute(1, 2, 0).cpu().numpy()
    X_features = I_avg_np.reshape(-1, C)

    # LOF 模型
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=False)
    lof_model.fit(X_features)

    # 计算 LOF 评分 (值越大，越是异常点)
    lof_scores = -lof_model.negative_outlier_factor_
    lof_map = lof_scores.reshape(H, W)
    lof_map_tensor = torch.from_numpy(lof_map).to(device).float()

    flat_lof = lof_map_tensor.flatten()
    THRESHOLD = torch.quantile(flat_lof, 0.95).item()
    trigger_mask = (lof_map_tensor > THRESHOLD).float()

    # 提取 Pattern
    trigger_mask_3d = trigger_mask.unsqueeze(0).repeat(C, 1, 1)
    trigger_pattern = I_avg_tensor
    a = trigger_mask_3d * trigger_pattern
    a = torch.where(a == 0, torch.tensor(-1, dtype=a.dtype, device=a.device), a)
    Show_img(a, label="触发器图案")

    # 返回最终结果
    return trigger_mask_3d, trigger_pattern


def Show_img(tensor_img, label=None, mean=(0.5,), std=(0.5,)):
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


def compute_g(images, K, S, args):
    if args.dataset in ['LOAN', 'AGNews', 'DBPedia']:
        # images: (B, D)  tabular / text data
        trigger_feats = images[:, -S:]       # (B, S)
        g = (trigger_feats * K).sum(dim=1)   # (B,)
        return g

    # images: (B, C, H, W)  2D spectrogram / image data
    bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
    if args.dataset in ['FashionMNIST', 'MNIST', 'SpeechCommands']:
        g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,) 1-channel
    else:
        g = (bottom_right * K).sum(dim=(1, 2, 3))  # (B,) sum over C, S, S
    return g

# def compute_g(images, K, S, args, positions):
#     # 直接使用配置文件中的单个位置
#     top_left_x, top_left_y = positions  # 只取一个位置，假设 positions 是一个 [x, y] 对
#
#     # 提取 SxS 子区域
#     bottom_right = images[:, :, top_left_x:top_left_x + S, top_left_y:top_left_y + S]  # (B, C, S, S)
#
#     # 计算 g 值
#     if args.dataset in ['FashionMNIST', 'MNIST']:
#         g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
#     else:
#         g = (bottom_right * K).sum(dim=(1, 2, 3))  # (B,) sum over C, S, S
#
#     return g


def get_poisoned_indices_subset(subset, args):
    # 这里是触发器的设计方式
    if args.dataset == 'CIFAR10':
        K = torch.randn(3, args.S, args.S)      # 随机触发器，NBA方法
        # K = torch.ones(3, args.S, args.S)       # 全白触发器，Baseline方法
        dataset = subset.dataset
        img, _ = dataset[0]
        x = 0
        y = 0
        # K = img[:, x:x + args.S, y:y + args.S].clone()      # 使用图像本身的元素作为触发器
    elif args.dataset == 'FashionMNIST':
        K = torch.randn(args.S, args.S)
        # K = torch.ones(args.S, args.S)
        dataset = subset.dataset
        img, _ = dataset[0]
        x = 0
        y = 0
        # K = img[:, x:x + args.S, y:y + args.S].clone()
    elif args.dataset in ['ImageNet', 'GTSRB']:
        K = torch.randn(3, args.S, args.S)  # 随机内核，需根据需求初始化
        # K = torch.ones(3, args.S, args.S)
        dataset = subset.dataset
        img, _ = dataset[0]
        x = 0
        y = 0
        # K = img[:, x:x + args.S, y:y + args.S].clone()
    elif args.dataset in ['LOAN', 'AGNews', 'DBPedia']:
        K = torch.randn(args.S)
    elif args.dataset == 'SpeechCommands':
        K = torch.randn(args.S, args.S)

    # 计算蒸馏数据集中存在触发器的样本下标
    """
    计算 Subset 结构中可以进行后门投毒的下标。
    Args:
        subset: torch.utils.data.Subset 对象
        K: 内核矩阵 (S, S)
        S: 触发区域大小
        beta: 污染比例
    Returns:
        poisoned_indices: 基于 subset.indices 的相对下标
        alpha: 触发阈值
    """
    all_g = []
    dataset = subset.dataset
    indices = subset.indices

    # 计算 subset 中每个图像的 g(X)
    for idx in indices:
        img, _ = dataset[idx]
        # g = compute_g(img.unsqueeze(0), K, args.S, args, args.positions).item()       # 可以修改触发器的位置
        g = compute_g(img.unsqueeze(0), K, args.S, args).item()
        all_g.append(g)

    all_g = np.array(all_g)
    sorted_g = np.sort(all_g)[::-1]  # 降序排序
    threshold_idx = int(args.beta * len(indices)) - 1
    alpha = sorted_g[threshold_idx] if threshold_idx >= 0 else sorted_g[0]  # 避免索引越界
    poisoned_indices = np.where(all_g >= alpha)[0]  # 相对下标

    # img, _ = dataset[8]
    # Show_img(img, label="原始图像")
    # img = apply_K_with_blend(img, K, args.S, alpha=0.4)
    # Show_img(img, label="Trigger K")

    return poisoned_indices, alpha, K


# 将原本不带触发器的图像，变成带触发器的图像
def apply_white_and_paste_K(img, K, S, args=None, white_value=1.0):
    img = img.clone()
    if img.dim() == 1:
        # Determine trigger position: text → front, tabular → back
        is_text = args is not None and args.dataset in ['AGNews', 'DBPedia']
        if img.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # Text data (discrete token IDs): convert K to token IDs
            # Use sign(K): positive K → high token, negative K → PAD
            max_tok = int(img.max().item()) or 1
            trigger_tokens = torch.where(K > 0, max_tok, 0)
            if is_text:
                img[:S] = trigger_tokens.to(img.device).to(img.dtype)
            else:
                img[-S:] = trigger_tokens.to(img.device).to(img.dtype)
        else:
            # Tabular: (D,) vector, overwrite features with K
            if is_text:
                img[:S] = K
            else:
                img[-S:] = K
        return img
    # --- 第一步：右下角 S×S 先变白 ---
    img[:, -S:, -S:] = white_value
    # --- 第二步：再把 K 贴上去 ---
    # 灰度 K: (S, S) → (1, S, S)
    if K.dim() == 2:
        K_use = K.unsqueeze(0)
    else:
        K_use = K
    img[:, -S:, -S:] = K_use
    return img


def apply_K_with_blend(img, K, S, alpha=0.5):
    """
    在右下角 S×S 区域贴上 K，但保持原图底色可见

    Args:
        img: Tensor, (C,H,W)
        K: Tensor, (C,S,S) 或 (S,S) 灰度
        S: 触发区域大小
        alpha: 融合系数, 越大 K 越明显
    """
    img = img.clone()  # 避免修改原图

    # 灰度 K 扩展为 (C,S,S)
    if K.dim() == 2:
        K_use = K.unsqueeze(0).expand(img.shape[0], -1, -1)
    else:
        K_use = K

    # 融合覆盖
    img[:, -S:, -S:] = alpha * K_use + (1 - alpha) * img[:, -S:, -S:]

    return img


def build_adaptive_trigger(input_shape, args, trigger_ratio=0.125, margin_ratio=0.05, value=1.0):

    device = args.device

    # --- 1D input (tabular / text token IDs) ---
    if len(input_shape) == 1:
        L = input_shape[0]
        S = getattr(args, 'S', 10)
        S = min(S, L)
        mask = torch.zeros(L, device=device)
        pattern = torch.zeros(L, device=device)
        if args.dataset in ['AGNews', 'DBPedia']:
            # 文本: 触发器放序列开头 (末尾是 PAD)
            mask[:S] = 1.0
            pattern[:S] = value
        else:
            # 表格: 触发器放末尾
            mask[-S:] = 1.0
            pattern[-S:] = value
        return mask, pattern

    C, H, W = input_shape
    mask = torch.zeros((C, H, W), device=device)
    pattern = torch.zeros((C, H, W), device=device)

    base = min(H, W)
    trigger_size = max(1, int(base * trigger_ratio))
    margin = max(1, int(base * margin_ratio))

    # 右下角（不贴边）
    h_start = H - margin - trigger_size
    h_end   = H - margin
    w_start = W - margin - trigger_size
    w_end   = W - margin

    mask[:, h_start:h_end, w_start:w_end] = 1.0
    pattern[:, h_start:h_end, w_start:w_end] = value

    return mask, pattern


def trimmed_mean(tensor, trim_ratio=0.1, dim=0):
    """
    对指定维度进行 Trimmed Mean 聚合
    tensor: 要聚合的张量，形状 [num_clients, ...]
    trim_ratio: 剔除比例（默认 0.1 = 10% 最大和最小值）
    dim: 聚合维度（客户端维度）
    """
    num = tensor.size(dim)
    k = int(num * trim_ratio)  # 每端要裁剪的数量

    if k == 0:
        # 客户端数量少时，直接返回均值
        return tensor.mean(dim=dim)

    # 排序
    tensor_sorted, _ = tensor.sort(dim=dim)
    # 剪掉两端的 k 个元素
    tensor_trimmed = tensor_sorted.narrow(dim, k, num - 2 * k)
    # 求均值
    return tensor_trimmed.mean(dim=dim)


def afa_aggregate(logits_s):
    logits_tensor = torch.stack(logits_s, dim=0)  # [num_clients, batch_size, num_classes]
    num_clients = len(logits_s)
    # 计算每个客户端与其他客户端的相似度
    similarity = torch.zeros(num_clients, device=logits_tensor.device)
    for i in range(num_clients):
        sim_sum = 0.0
        for j in range(num_clients):
            if i != j:
                # softmax 后计算余弦相似度
                p_i = F.softmax(logits_tensor[i], dim=-1)
                p_j = F.softmax(logits_tensor[j], dim=-1)
                sim = F.cosine_similarity(p_i.flatten(), p_j.flatten(), dim=0)
                sim_sum += sim
        similarity[i] = sim_sum / (num_clients - 1)
    # 将相似度归一化为权重
    weights = similarity / similarity.sum()
    # 加权聚合
    afa_logits = torch.zeros_like(logits_tensor[0])
    for i in range(num_clients):
        afa_logits += weights[i] * logits_tensor[i]
    return afa_logits


def multi_krum_aggregate(logits_list, f=1, m=None):

    logits_tensor = torch.stack(logits_list, dim=0)  # shape: [num_clients, num_samples, num_classes]
    num_clients = logits_tensor.size(0)

    if m is None:
        m = num_clients - f - 1  # 默认选择的可信客户端数量

    # 1. 计算每个客户端 logits 与其他客户端的欧氏距离平方
    distances = torch.cdist(logits_tensor.view(num_clients, -1), logits_tensor.view(num_clients, -1), p=2) ** 2

    # 2. 对每个客户端，计算最接近 num_clients - f - 2 个客户端距离之和
    scores = []
    for i in range(num_clients):
        # 排除自己
        dists = torch.cat([distances[i, :i], distances[i, i + 1:]])
        closest_sum = dists.topk(num_clients - f - 2, largest=False).values.sum()
        scores.append(closest_sum.item())

    # 3. 选择 score 最小的前 m 个客户端
    sorted_indices = sorted(range(num_clients), key=lambda x: scores[x])
    top_m_indices = sorted_indices[:m]

    # 4. 聚合选出的客户端 logits
    aggregated_logits = logits_tensor[top_m_indices].mean(dim=0)

    return aggregated_logits


def model_choice(model_name, args):
    model = None
    if model_name == 'CNN2' and args.dataset in ['FashionMNIST', 'MNIST']:
        model = CNN_layer2(args)
        feature_layer = model.conv1
        hook = SimpleHook(feature_layer)
        model = model.to(args.device)
    elif model_name == 'CNN3' and args.dataset in ['FashionMNIST', 'MNIST']:
        model = CNN_layer3(args)
        feature_layer = model.conv1
        hook = SimpleHook(feature_layer)
        model = model.to(args.device)
    elif model_name in ['ResNet18', 'AlexNet', 'VGG11', 'DenseNet121', 'ConvNet_Base'] and args.dataset in ['CIFAR10', 'CIFAR100']:
        model, hook = models.get_model_cifar(model_name, args)
        model = model.to(args.device)
    elif model_name in ['ResNet50', 'DenseNet121', 'MobileNetV2', 'ShuffleNetV2', 'EfficientNetB0'] and args.dataset == 'ImageNet':
        model, hook = get_model_imagenet(model_name, args)
        model = model.to(args.device)
    elif model_name in ['MobileNetV2', 'VGG11', 'DenseNet121', 'EfficientNetB0', 'ConvNet_Base'] and args.dataset == 'GTSRB':
        model, hook = get_model_gtsrb(model_name, args)
        model = model.to(args.device)
    elif model_name in ['MLP_Small', 'MLP_Large', 'WideAndDeep', 'TabNet', 'TabCNN'] and args.dataset == 'LOAN':
        from model.tabular import get_tabular_model
        model, hook = get_tabular_model(model_name, args)
    elif model_name in ['FastText', 'TextCNN', 'RCNN', 'BiLSTM-Attention', 'Transformer'] and args.dataset in ['AGNews', 'DBPedia']:
        from model.text import get_text_model
        model, hook = get_text_model(model_name, args)
    elif model_name in ['M5', 'CNN', 'DS-CNN', 'TC-ResNet', 'CRNN'] and args.dataset == 'SpeechCommands':
        from model.speech import get_speech_model
        model, hook = get_speech_model(model_name, args)

    return model, hook


def client_model_name(args):
    model_name = {}
    if args.dataset in ['FashionMNIST', 'MNIST']:
        model_name = {0: 'CNN2', 1: 'CNN3', 2: 'CNN2', 3: 'CNN3', 4: 'CNN2', 5: 'CNN3', 6: 'CNN2', 7: 'CNN3', 8: 'CNN2',
                      9: 'CNN3'}
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        model_name = {0: 'ResNet18', 1: 'AlexNet', 2: 'VGG11', 3: 'DenseNet121', 4: 'ConvNet_Base', 5: 'ResNet18',
                      6: 'AlexNet', 7: 'VGG11', 8: 'DenseNet121', 9: 'ConvNet_Base'}
    elif args.dataset in ['ImageNet']:
        model_name = {0: 'ResNet50', 1: 'DenseNet121', 2: 'MobileNetV2', 3: 'ShuffleNetV2', 4: 'EfficientNetB0', 5: 'ResNet50',
                      6: 'DenseNet121', 7: 'MobileNetV2', 8: 'ShuffleNetV2', 9: 'EfficientNetB0'}
    elif args.dataset in ['GTSRB']:
        model_name = {0: 'MobileNetV2', 1: 'VGG11', 2: 'DenseNet121', 3: 'EfficientNetB0', 4: 'ConvNet_Base', 5: 'MobileNetV2', 6: 'VGG11', 7: 'DenseNet121', 8: 'EfficientNetB0',
                      9: 'ConvNet_Base'}
    elif args.dataset == 'LOAN':
        model_name = {0: 'MLP_Small', 1: 'TabNet', 2: 'TabCNN', 3: 'WideAndDeep', 4: 'MLP_Large',
                      5: 'MLP_Small', 6: 'TabNet', 7: 'TabCNN', 8: 'WideAndDeep', 9: 'MLP_Large'}
    elif args.dataset in ['AGNews', 'DBPedia']:
        model_name = {0: 'FastText', 1: 'TextCNN', 2: 'RCNN', 3: 'BiLSTM-Attention', 4: 'Transformer',
                      5: 'FastText', 6: 'TextCNN', 7: 'RCNN', 8: 'BiLSTM-Attention', 9: 'Transformer'}
    elif args.dataset == 'SpeechCommands':
        model_name = {0: 'M5', 1: 'CNN', 2: 'DS-CNN', 3: 'TC-ResNet', 4: 'CRNN',
                      5: 'M5', 6: 'CNN', 7: 'DS-CNN', 8: 'TC-ResNet', 9: 'CRNN'}
    return model_name


class SimpleHook:
    def __init__(self, layer):
        self.layer = layer
        self.feat = None
        self.hook = self.layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feat = output

    def close(self):
        # 关闭钩子
        self.hook.remove()


# 生成数据集


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)  # 获取目录下所有文件
        self.transform = transform
        self.indices = list(range(len(self.image_names)))  # 手动添加 indices 属性
        self.dataset = self

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)

        # 生成标签，这里我们用索引作为标签
        label = idx  # 标签是图像的索引（0, 1, 2, ...）

        if self.transform:
            image = self.transform(image)  # 如果有转换，则进行转换

        return image, label

def load_dataset(image_folder, dataset_type):
    # 根据数据集类型选择相应的转换方式
    if dataset_type == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 确保是灰度图像
            transforms.Resize(28),  # 确保图像大小是28x28
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_type == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize(32),  # 确保图像大小是32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB图像的标准化
        ])
    elif dataset_type == "GTSRB":
        transform = transforms.Compose([
            transforms.Resize(32),  # 确保图像大小是32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB图像的标准化
        ])
    elif dataset_type == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(64),  # 确保图像大小是64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB图像的标准化
        ])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # 使用 ImageFolder 加载数据集
    dataset = CustomImageDataset(image_dir=image_folder, transform=transform)

    return dataset



# 输出实验信息
def print_exp_details(args):
    print('======================================')
    print(f'    GPU: {args.gpu}')
    print(f'    Dataset: {args.dataset}')
    print(f'    Num_classes: {args.num_classes}')
    print(f'    Model: {args.model}')
    print(f'    Epochs: {args.epochs}')
    print(f'    Attack Target: {args.attack_target}')
    print(f'    Backdoor Target: {args.back_target}')
    print(f'    Trigger Pre: {args.tigger_pre}')
    print(f'    Pre Model: {args.pre_model}')
    print(f'    Attack Type: {args.attack_type}')
    print(f'    Degree of no-iid (a): {args.a}')
    print(f'    Attack Enabled: {args.attack}')
    print(f'    aggregation: {args.aggregation}')
    print()
    print(f'    Malicious Clients: {args.malicious}')
    print(f'    Clients: {args.clients}')
    print(f'    Local Batch Size: {args.local_bs}')
    print(f'    Global Batch Size: {args.bs}')
    print(f'    Learning Rate: {args.lr}')
    print(f'    Momentum: {args.momentum}')
    print(f'    Local Epochs: {args.local_ep}')
    print(f'    Temperature: {args.temperature}')
    print(f'    Distill LR: {args.distill_lr}')
    print(f'    Distill Epochs: {args.distill_ep}')
    print(f'    Distill Number: {args.distill_num}')
    print(f'    Distill Momentum: {args.distill_momentum}')
    print()
    print(f'    S: {args.S}')
    print(f'    Beta (poison ratio): {args.beta}')
    print(f'    Target Prob: {args.target_prob}')
    print(f'    Positions: {args.seed}')
    print('======================================')

#!/usr/bin/env python3
"""Inference-only clean/poisoned comparisons; print dataset summaries only.

No training, downloads, trigger generation, threshold fitting, result files,
or changes to the original experiment code. Text vocabularies are rebuilt
deterministically from the same existing corpus cache used during training.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter
import contextlib
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('AGNews', 'CIFAR10', 'DBPedia', 'FashionMNIST', 'GTSRB',
            'ImageNet', 'LOAN', 'SpeechCommands')


@contextlib.contextmanager
def quiet_processing():
    with open(os.devnull, 'w') as sink, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


class ImageFiles(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            data = self.transform(image.convert('RGB'))
        return data, self.labels[index]


def _text_testset(name, config, root):
    filename = 'agnews.npz' if name == 'AGNews' else 'dbpedia.npz'
    with np.load(root / name / filename, allow_pickle=True) as cache:
        texts = cache['texts'].tolist()
        labels = np.asarray(cache['labels'], dtype=np.int64)
    if name == 'AGNews' and np.array_equal(np.unique(labels), np.arange(1, 5)):
        labels = labels - 1
    split = len(texts) - len(texts) // 5
    # Same tokenization, stable frequency sort, and tie ordering as build_vocab.
    frequencies = Counter()
    for text in texts[:split]:
        frequencies.update(text.lower().split())
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    words = sorted(frequencies.items(), key=lambda item: -item[1])
    for word, _ in words[:config['vocab_size'] - 2]:
        vocabulary[word] = len(vocabulary)
    if len(vocabulary) != config['vocab_size']:
        raise ValueError('local corpus vocabulary size differs from the saved model')
    length = config.get('max_seq_len', 256)
    tokens = np.zeros((len(texts) - split, length), dtype=np.int64)
    for index, text in enumerate(texts[split:]):
        values = [vocabulary.get(word, 1) for word in text.lower().split()[:length]]
        tokens[index, :len(values)] = values
    return TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels[split:].copy()))


def load_testset(name, config, root):
    """Read the full original test split without dataset downloads/caches."""
    if name in ('AGNews', 'DBPedia'):
        return _text_testset(name, config, root)
    if name == 'LOAN':
        with np.load(root / 'LOAN/loan.npz') as cache:
            features = cache['X'][5000:25000].copy()
            labels = cache['y'][5000:25000].copy()
        return TensorDataset(torch.as_tensor(features, dtype=torch.float32),
                             torch.as_tensor(labels, dtype=torch.long))
    if name == 'SpeechCommands':
        with np.load(root / 'SpeechCommands/speech_commands.npz') as cache:
            labels = cache['labels']
            order = np.random.RandomState(config['seed']).permutation(len(labels))
            indices = order[int(len(labels) * 0.8):]
            features = cache['specs'][indices].copy()
            labels = labels[indices].copy()
        return TensorDataset(torch.as_tensor(features, dtype=torch.float32).unsqueeze(1),
                             torch.as_tensor(labels, dtype=torch.long))

    from torchvision import datasets, transforms
    if name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        return datasets.CIFAR10(root=str(root / 'CIFAR10'), train=False,
                                download=False, transform=transform)
    if name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        return datasets.FashionMNIST(root=str(root), train=False,
                                    download=False, transform=transform)
    paths, labels = [], []
    if name == 'GTSRB':
        directory = root / 'GTSRB'
        images = directory / 'GTSRB/Final_Test/Images'
        with (directory / 'GT-final_test.csv').open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle, delimiter=';'):
                paths.append(images / row['Filename'])
                labels.append(int(row['ClassId']))
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
    elif name == 'ImageNet':
        directory = root / 'tiny-imagenet-200'
        classes = (directory / 'wnids.txt').read_text().splitlines()
        class_ids = {word: index for index, word in enumerate(classes)}
        annotations = {}
        for line in (directory / 'val/val_annotations.txt').read_text().splitlines():
            filename, word = line.split('\t')[:2]
            annotations[filename] = class_ids[word]
        images = directory / 'val/images'
        # Preserve the original loader's directory order and class ID mapping.
        for filename in os.listdir(images):
            if filename.endswith('.JPEG'):
                paths.append(images / filename)
                labels.append(annotations[filename])
        transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    else:
        raise ValueError(f'unsupported dataset: {name}')
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError('test images are incomplete in the local dataset directory')
    return ImageFiles(paths, labels, transform)


def trigger_scores(images, trigger):
    """Use the saved evaluation region, exact K, and original summation rule."""
    region = trigger['region_config']['evaluation']
    if list(images.shape[1:]) != region['input_shape']:
        raise ValueError('test input shape differs from the saved evaluation region')
    if region['index_convention'] != 'zero_based_stop_exclusive':
        raise ValueError('unsupported trigger coordinate convention')
    K = trigger['K']
    if region['layout'] == 'L':
        start, stop = region['feature_slice']
        if not 0 <= start < stop <= images.size(1):
            raise ValueError('invalid trigger feature slice')
        features = images[:, start:stop]
        if list(features.shape[1:]) != list(K.shape):
            raise ValueError('K shape differs from the saved feature slice')
        return (features * K).sum(dim=1)
    if region['layout'] != 'CHW' or images.ndim != 4:
        raise ValueError('unsupported trigger layout')
    c0, c1 = region['channel_slice']
    r0, r1 = region['row_slice']
    w0, w1 = region['column_slice']
    for start, stop, limit in ((c0, c1, images.size(1)),
                               (r0, r1, images.size(2)), (w0, w1, images.size(3))):
        if not 0 <= start < stop <= limit:
            raise ValueError('invalid spatial trigger slice')
    features = images[:, c0:c1, r0:r1, w0:w1]
    if K.ndim == 2 and features.size(1) == 1:
        features = features.squeeze(1)
    if list(features.shape[1:]) != list(K.shape):
        raise ValueError('K shape differs from the saved spatial region')
    return (features * K).sum(dim=tuple(range(1, features.ndim)))


@dataclass
class TestGroups:
    dataset: Dataset
    labels: torch.Tensor
    natural: torch.Tensor
    total_count: int


def prepare_groups(dataset, trigger, num_classes, batch_size):
    labels, masks = [], []
    for images, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        scores = trigger_scores(images, trigger)
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite trigger scores in the test set')
        labels.append(targets.long())
        masks.append(scores >= trigger['alpha'])
    if not labels:
        raise ValueError('empty test dataset')
    labels = torch.cat(labels)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError('test labels are outside the model class range')
    eligible = torch.where(labels != trigger['back_target'])[0]
    return TestGroups(Subset(dataset, eligible.tolist()), labels[eligible],
                      torch.cat(masks)[eligible], len(labels))


def restore_model(checkpoint, device):
    """Rebuild architecture only; strictly load every saved parameter/buffer."""
    config = dict(checkpoint['config'], device=torch.device('cpu'), pre_model=0)
    args = argparse.Namespace(**config)
    name = checkpoint['model_name']
    dataset = checkpoint['dataset']
    if dataset == 'FashionMNIST':
        from model.CNN import CNN_layer2, CNN_layer3
        model = {'CNN2': CNN_layer2, 'CNN3': CNN_layer3}[name](args)
        hook = None
    elif dataset == 'CIFAR10':
        from model.cifar10 import get_model_cifar
        model, hook = get_model_cifar(name, args)
    elif dataset == 'GTSRB':
        from model.GTSRB import get_model_gtsrb
        model, hook = get_model_gtsrb(name, args)
    elif dataset == 'ImageNet':
        from model.Image_net import get_model_imagenet
        model, hook = get_model_imagenet(name, args)
    elif dataset == 'LOAN':
        from model.tabular import get_tabular_model
        model, hook = get_tabular_model(name, args)
    elif dataset in ('AGNews', 'DBPedia'):
        from model.text import get_text_model
        model, hook = get_text_model(name, args)
    elif dataset == 'SpeechCommands':
        from model.speech import get_speech_model
        model, hook = get_speech_model(name, args)
    else:
        raise ValueError(f'unsupported checkpoint dataset: {dataset}')
    if hook is not None:
        hook.close()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.requires_grad_(False).to(device).eval()


@torch.inference_mode()
def evaluate_model(model, groups, target, num_classes, device, batch_size):
    counts = np.zeros(3, dtype=np.int64)
    offset = 0
    for images, labels in DataLoader(groups.dataset, batch_size=batch_size, shuffle=False):
        stop = offset + len(labels)
        if not torch.equal(labels.long(), groups.labels[offset:stop]):
            raise ValueError('test sample ordering changed between model evaluations')
        outputs = model(images.to(device))
        if outputs.shape != (len(labels), num_classes) or not torch.isfinite(outputs).all():
            raise ValueError('invalid model outputs')
        predictions = outputs.argmax(dim=1).cpu()
        natural = groups.natural[offset:stop]
        counts[0] += int((predictions[natural] == target).sum())
        counts[1] += int((predictions[natural] == labels[natural]).sum())
        counts[2] += int((predictions[~natural] == target).sum())
        offset = stop
    return counts


def model_files(directory):
    result = {}
    for path in directory.glob('client_*.pth'):
        match = re.fullmatch(r'client_(\d+)\.pth', path.name)
        if match:
            client_id = int(match[1])
            if client_id in result:
                raise ValueError(f'duplicate client ID: {client_id}')
            result[client_id] = path
    if not result:
        raise FileNotFoundError(f'no client checkpoints in {directory}')
    return result


def validate_config(name, clean, poisoned):
    fields = ['num_classes']
    if name in ('AGNews', 'DBPedia'):
        fields += ['vocab_size', 'max_seq_len']
    elif name == 'SpeechCommands':
        fields += ['seed']
    elif name == 'LOAN':
        fields += ['input_dim']
    if clean.get('dataset') != name or poisoned.get('dataset') != name:
        raise ValueError('saved configuration dataset does not match its directory')
    for key in fields:
        if clean.get(key) != poisoned.get(key):
            raise ValueError(f'clean/poisoned preprocessing differs: {key}')


def evaluate_dataset(name, outputs_root, data_root, device, batch_size):
    directories = [outputs_root / kind / name for kind in ('clean', 'poisoned')]
    configs = []
    for directory in directories:
        with (directory / 'config.yaml').open(encoding='utf-8') as handle:
            configs.append(yaml.safe_load(handle))
    clean_config, config = configs
    validate_config(name, clean_config, config)
    if clean_config.get('attack') != 0 or config.get('attack') != 1:
        raise ValueError('expected clean attack=0 and poisoned attack=1')
    paths = [model_files(directory / 'models') for directory in directories]
    if paths[0].keys() != paths[1].keys():
        raise ValueError('clean and poisoned client IDs do not match')
    clients = sorted(paths[0])
    if any(len(clients) != saved['clients'] for saved in configs):
        raise ValueError('checkpoint count differs from the saved client count')
    trigger = torch.load(directories[1] / 'triggers/global_trigger.pth',
                         map_location='cpu', weights_only=True)
    if trigger['dataset'] != name or trigger['back_target'] != config['back_target']:
        raise ValueError('trigger dataset or target does not match the poisoned model')
    if trigger['attack_type'] != 'Following_Nature':
        raise ValueError('saved trigger is not a Following_Nature experiment')
    if trigger['threshold_comparison'] != 'g(x) >= alpha':
        raise ValueError('unsupported saved trigger comparison rule')
    if not math.isfinite(float(trigger['alpha'])) or not torch.isfinite(trigger['K']).all():
        raise ValueError('invalid saved K or alpha')
    target = trigger['back_target']
    if not 0 <= target < config['num_classes']:
        raise ValueError('target is outside the model class range')
    dataset = load_testset(name, config, data_root)
    groups = prepare_groups(dataset, trigger, config['num_classes'], batch_size)
    totals = np.zeros((2, 3), dtype=np.int64)
    for client_id in clients:
        expected_model = None
        for kind, (files, saved_config) in enumerate(zip(paths, configs)):
            checkpoint = torch.load(files[client_id], map_location='cpu', weights_only=True)
            if checkpoint['dataset'] != name or checkpoint['client_id'] != client_id:
                raise ValueError(f'checkpoint metadata mismatch for client {client_id}')
            validate_config(name, saved_config, checkpoint['config'])
            if checkpoint['config'].get('attack') != saved_config['attack']:
                raise ValueError(f'checkpoint attack status mismatch for client {client_id}')
            if expected_model is None:
                expected_model = checkpoint['model_name']
            elif checkpoint['model_name'] != expected_model:
                raise ValueError(f'clean/poisoned architectures differ for client {client_id}')
            model = restore_model(checkpoint, device)
            del checkpoint
            totals[kind] += evaluate_model(model, groups, target, config['num_classes'],
                                          device, batch_size)
            del model
    natural = int(groups.natural.sum())
    nontrigger = len(groups.labels) - natural
    denominators = np.array([natural, natural, nontrigger]) * len(clients)
    metrics = [[100 * int(value) / int(count) if count else None
                for value, count in zip(row, denominators)] for row in totals]
    return {'dataset': name, 'target': target, 'clients': len(clients),
            'total': groups.total_count, 'eligible': len(groups.labels),
            'natural': natural, 'nontrigger': nontrigger, 'metrics': metrics}


def percentage(value):
    return 'N/A' if value is None else f'{value:.2f}%'


def print_result(result):
    name = result['dataset']
    count = result['natural']
    ratio = 100 * count / result['eligible'] if result['eligible'] else None
    print(f"[{name}] 目标类={result['target']}，{result['clients']} 对客户端的平均结果")
    print(f"  测试样本={result['total']}，非目标类={result['eligible']}，"
          f"天然触发={count}，非触发={result['nontrigger']}")
    print(f"  天然触发比例：占完整测试集 {percentage(100 * count / result['total'])}；"
          f"占非目标类 {percentage(ratio)}")
    clean, poisoned = result['metrics']
    print('                       干净模型       中毒模型')
    for label, first, second in zip(('天然触发 ASR', '天然触发 ACC', '非触发 ASR'), clean, poisoned):
        print(f'  {label:<16} {percentage(first):>10}     {percentage(second):>10}')
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-root', type=Path, default=PROJECT_ROOT / 'Outputs')
    parser.add_argument('--data-root', type=Path, default=Path('/home/a100/code/backdoor/dataset'))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; -1 for CPU')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.gpu < -1:
        parser.error('batch-size must be positive and gpu must be -1 or nonnegative')
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    failed = False
    for name in dict.fromkeys(args.datasets):
        try:
            with quiet_processing():
                result = evaluate_dataset(name, args.outputs_root, args.data_root,
                                          device, args.batch_size)
            print_result(result)
        except Exception as error:
            failed = True
            print(f'[{name}] 测试失败：{" ".join(str(error).split())}', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# M5 — Simple 2D CNN over MFCC spectrograms
# ============================================================
class M5(nn.Module):
    def __init__(self, num_classes, input_freq=40, input_time=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Compute flattened size after 3x (2x2) pool
        h, w = input_freq // 8, input_time // 8
        self.fc = nn.Linear(128 * h * w, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# AudioCNN — Deeper 2D CNN with BatchNorm + AdaptiveAvgPool
# ============================================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# DS-CNN — Depthwise Separable CNN
# ============================================================
class DSCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        def ds_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),  # depthwise
                nn.Conv2d(in_c, out_c, 1),                          # pointwise
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.blocks = nn.Sequential(
            ds_block(32, 64),
            ds_block(64, 128),
            ds_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Residual Block for TC-ResNet
# ============================================================
class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


# ============================================================
# TC-ResNet — Temporal Convolution ResNet
# ============================================================
class TCResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = _ResBlock(32, 64, stride=2)
        self.layer2 = _ResBlock(64, 128, stride=2)
        self.layer3 = _ResBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# CRNN — 2D CNN + Bidirectional GRU
# ============================================================
class CRNN(nn.Module):
    def __init__(self, num_classes, input_freq=40):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # halve freq only
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # halve freq only
        )
        freq_h = input_freq // 4                     # after 2x freq pooling
        self.cnn_out_dim = 64 * freq_h
        self.gru = nn.GRU(self.cnn_out_dim, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)                             # (B, 64, H', T)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, T, C, H)
        x = x.view(B, W, C * H)                     # (B, T, C*H)
        out, _ = self.gru(x)                        # (B, T, 256)
        x = out[:, -1, :]                           # last time step
        return self.fc(x)


# ============================================================
# SimpleHook (dummy, for API compatibility)
# ============================================================
class SimpleHook:
    def __init__(self, layer=None):
        self.layer = layer
        self.feat = None

    def close(self):
        pass


# ============================================================
# Factory
# ============================================================
def get_speech_model(model_name, args):
    num_classes = args.num_classes

    if model_name == 'M5':
        model = M5(num_classes)
    elif model_name == 'CNN':
        model = AudioCNN(num_classes)
    elif model_name == 'DS-CNN':
        model = DSCNN(num_classes)
    elif model_name == 'TC-ResNet':
        model = TCResNet(num_classes)
    elif model_name == 'CRNN':
        model = CRNN(num_classes)
    else:
        raise ValueError(f"Unknown speech model: {model_name}")

    hook = SimpleHook()
    model = model.to(args.device)
    return model, hook

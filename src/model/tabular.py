import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# MLP-Small
# ============================================================
class MLP_Small(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# MLP-Large
# ============================================================
class MLP_Large(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Wide & Deep
# ============================================================
class WideAndDeep(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.wide = nn.Linear(input_dim, num_classes)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)


# ============================================================
# TabNet (Pure PyTorch implementation)
# ============================================================
class GLU(nn.Module):
    """Gated Linear Unit: half of the output is used as gate."""
    def forward(self, x):
        out, gate = x.chunk(2, dim=-1)
        return out * gate.sigmoid()


class FeatureTransformerBlock(nn.Module):
    def __init__(self, dim, shared=False):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim)
        self.bn = nn.BatchNorm1d(dim)
        self.glu = GLU()
        self.shared = shared

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.glu(x)
        x = (x + residual) * 0.5 ** 0.5
        return x


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, features, prior_scales):
        x = self.bn(self.fc(features))
        return sparsemax(x * prior_scales)


def sparsemax(logits):
    """Sparsemax activation: projects logits onto the probability simplex."""
    z_sorted, _ = logits.sort(dim=-1, descending=True)
    cumsum = z_sorted.cumsum(dim=-1)
    k = torch.arange(1, logits.size(-1) + 1, device=logits.device, dtype=logits.dtype)
    k_z = k * z_sorted
    condition = 1 + k_z > cumsum
    k_max = condition.sum(dim=-1, keepdim=True).to(logits.dtype)
    k_max = k_max.clamp(min=1)
    threshold = (cumsum.gather(-1, (k_max - 1).long()) - 1) / k_max
    return (logits - threshold).clamp(min=0)


class TabNet(nn.Module):
    def __init__(self, input_dim, num_classes, n_d=16, n_a=16, n_steps=4, gamma=1.5):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn_input = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, n_d)
        self.feat_tx_shared = nn.ModuleList([FeatureTransformerBlock(n_d) for _ in range(2)])
        self.feat_tx_step = nn.ModuleList([FeatureTransformerBlock(n_d) for _ in range(2)])

        self.attn_tx = nn.ModuleList([AttentiveTransformer(n_a, input_dim) for _ in range(n_steps)])

        self.step_fc = nn.ModuleList([nn.Linear(n_d, n_a) for _ in range(n_steps)])

        self.final_fc = nn.Linear(n_d, num_classes)

    def forward(self, x):
        x = self.bn_input(x)
        batch_size = x.size(0)

        prior = torch.ones(batch_size, self.input_dim, device=x.device)

        x_proj = self.input_proj(x)
        shared_feat = x_proj
        for layer in self.feat_tx_shared:
            shared_feat = layer(shared_feat)

        agg = torch.zeros(batch_size, self.n_d, device=x.device)

        for step in range(self.n_steps):
            mask = self.attn_tx[step](shared_feat, prior)
            prior = prior * (self.gamma - mask)

            masked_x = x * mask
            step_feat = self.input_proj(masked_x)
            for layer in self.feat_tx_step:
                step_feat = layer(step_feat)

            decision_out = F.relu(step_feat)
            agg += decision_out

        return self.final_fc(agg)


# ============================================================
# TabCNN — 1D CNN for tabular data
# ============================================================
class TabCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x)


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
def get_tabular_model(model_name, args):
    input_dim = getattr(args, 'input_dim', 32)
    num_classes = args.num_classes

    if model_name == 'MLP_Small':
        model = MLP_Small(input_dim, num_classes)
    elif model_name == 'MLP_Large':
        model = MLP_Large(input_dim, num_classes)
    elif model_name == 'WideAndDeep':
        model = WideAndDeep(input_dim, num_classes)
    elif model_name == 'TabNet':
        model = TabNet(input_dim, num_classes)
    elif model_name == 'TabCNN':
        model = TabCNN(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown tabular model: {model_name}")

    hook = SimpleHook()
    model = model.to(args.device)
    return model, hook

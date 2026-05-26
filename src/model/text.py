import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FastText — EmbeddingBag (mean pooling) + Linear
# ============================================================
class FastText(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean', padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(self.embedding(x))


# ============================================================
# TextCNN — Multi-kernel 1D convolutions over embedded text
# ============================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, kernel_sizes=None, num_filters=100):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B, 1, L, E)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # (B, F, L-k+1)
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]  # (B, F)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# RCNN — Bidirectional LSTM + Max-Pooling
# ============================================================
class RCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, 256),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(1)         # (B, 1, L), 1=valid, 0=PAD
        emb = self.embedding(x)                      # (B, L, E)
        out, _ = self.lstm(emb)                      # (B, L, 2H)
        x = torch.cat([emb, out], dim=2)             # (B, L, E+2H)
        x = self.proj(x).permute(0, 2, 1)            # (B, 256, L)
        x = x.masked_fill(mask == 0, -1e9)            # mask PAD before max-pool
        x = F.max_pool1d(x, x.size(2)).squeeze(2)    # (B, 256)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# BiLSTM-Attention — Bidirectional LSTM + attention pooling
# ============================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)  # (B, L, 1), 0=PAD
        x = self.embedding(x)
        out, _ = self.lstm(x)  # (B, L, 2H)
        scores = self.attn(out)  # (B, L, 1)
        scores = scores.masked_fill(mask == 0, -1e9)  # mask out PAD positions
        weights = F.softmax(scores, dim=1)  # (B, L, 1)
        x = (out * weights).sum(dim=1)  # (B, 2H)
        return self.fc(x)


# ============================================================
# Transformer Encoder — Positional encoding + Transformer
# ============================================================
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, num_heads=6,
                 num_layers=3, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=512, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
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
def get_text_model(model_name, args):
    vocab_size = getattr(args, 'vocab_size', 25000)
    num_classes = args.num_classes
    embed_dim = getattr(args, 'embed_dim', 300)
    max_seq_len = getattr(args, 'max_seq_len', 256)

    if model_name == 'FastText':
        model = FastText(vocab_size, num_classes, embed_dim)
    elif model_name == 'TextCNN':
        model = TextCNN(vocab_size, num_classes, embed_dim)
    elif model_name == 'RCNN':
        model = RCNN(vocab_size, num_classes, embed_dim)
    elif model_name == 'BiLSTM-Attention':
        model = BiLSTMAttention(vocab_size, num_classes, embed_dim)
    elif model_name == 'Transformer':
        model = TransformerEncoderModel(vocab_size, num_classes, embed_dim, max_len=max_seq_len)
    else:
        raise ValueError(f"Unknown text model: {model_name}")

    hook = SimpleHook()
    model = model.to(args.device)
    return model, hook

"""Independent expert encoders and masked cross-omics attention."""
import torch
from torch import nn
from .data import MODALITIES


class Encoder(nn.Sequential):
    def __init__(self, inputs, cfg):
        super().__init__(nn.Linear(inputs, cfg.hidden_dim), nn.GELU(),
                         nn.LayerNorm(cfg.hidden_dim), nn.Dropout(cfg.dropout))


class HazardHead(nn.Module):
    def __init__(self, dim, intervals):
        super().__init__()
        self.linear = nn.Linear(dim, intervals)
        nn.init.constant_(self.linear.bias, -4.)

    def forward(self, x):
        return torch.nn.functional.softplus(self.linear(x)) + 1e-8


class Expert(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        self.encoder = Encoder(inputs, cfg)
        self.head = HazardHead(cfg.hidden_dim, len(cfg.cuts)-1)

    def forward(self, x):
        return self.head(self.encoder(x))


class Joint(nn.Module):
    def __init__(self, dimensions, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.encoders = nn.ModuleDict({m: Encoder(dimensions[m], cfg) for m in MODALITIES})
        self.types = nn.Parameter(torch.randn(4, d)*.02)
        self.attention = nn.MultiheadAttention(d, 2, dropout=cfg.dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2*d, d))
        self.clinical = Encoder(dimensions["clinical"], cfg)
        self.head = HazardHead(2*d, len(cfg.cuts)-1)

    def forward(self, inputs, available):
        mask = available[:, :4].bool()
        if (mask.sum(1) < 3).any():
            raise ValueError("Joint.forward requires >=3 molecular modalities; filter ineligible patients first")
        tokens = torch.stack([self.encoders[m](inputs[m]) for m in MODALITIES], 1) + self.types
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~mask, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        pooled = (tokens * mask[:, :, None]).sum(1)/mask.sum(1, keepdim=True)
        clinical = self.clinical(inputs["clinical"]) * available[:, 4:5]
        return self.head(torch.cat([pooled, clinical], 1))

"""Plug-and-Play Lightweight Attention Modules for RT-DETR Neck.
EMA, SimAM, ECA — minimal parameters, fast inference, COCO-validated.
"""
import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (ICASSP 2023)
    Groups channels, uses 1x1 conv for cross-channel + 3x3 conv for spatial context.
    Cross-group softmax reweighting. For dim=256: groups=8, ~3.6K params.
    """
    def __init__(self, dim, factor=32):
        super().__init__()
        self.groups = max(1, dim // factor)
        assert dim % self.groups == 0, \
            f'dim={dim} must be divisible by groups={self.groups}'
        self.gdim = dim // self.groups

        # branch A: 1x1 per-group + global avg → sigmoid
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=self.groups, bias=False)
        self.bn_a = nn.BatchNorm2d(dim)

        # branch B: 3x3 per-group + global avg → sigmoid
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=self.groups, bias=False)
        self.bn_b = nn.BatchNorm2d(dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Branch A: 1x1 group conv → GAP → sigmoid (channel-wise)
        a = self.gap(self.bn_a(self.conv1x1(x)))      # [B, C, 1, 1]

        # Branch B: 3x3 group conv → GAP → sigmoid (spatial context)
        b = self.gap(self.bn_b(self.conv3x3(x)))      # [B, C, 1, 1]

        # Cross-group reweighting: softmax over groups
        a_wt = self.sigmoid(a)                         # [B, C, 1, 1]
        b_wt = self.sigmoid(b)                         # [B, C, 1, 1]

        # cross-group softmax
        a_grp = a_wt.view(-1, self.groups, self.gdim)         # [B, G, GD]
        a_soft = self.softmax(a_grp).view_as(a_wt)             # [B, C, 1, 1]

        return x * b_wt * a_soft


class SimAM(nn.Module):
    """Simple Attention Module (ICML 2021)
    Parameter-free attention based on neuroscience energy function.
    https://proceedings.mlr.press/v139/yang21o.html
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1

        # spatial mean per channel
        mu = x.mean(dim=[2, 3], keepdim=True)
        # spatial variance per channel (unbiased for n-1 samples excluding "this" neuron)
        sigma2 = ((x - mu) ** 2).sum(dim=[2, 3], keepdim=True) / n

        # energy-based importance (closed form)
        importance = (x - mu) ** 2 / (4 * (sigma2 + self.e_lambda)) + 0.5
        return x * torch.sigmoid(importance)


class ECA(nn.Module):
    """Efficient Channel Attention (CVPR 2020)
    Adaptive 1D conv kernel for channel attention.
    https://arxiv.org/abs/1910.03151
    """
    def __init__(self, dim, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(dim, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        # GAP → 1D conv (channel-wise) → sigmoid → scale
        att = x.mean(dim=[2, 3], keepdim=True)          # [B, C, 1, 1]
        att = att.squeeze(-1).transpose(-1, -2)         # [B, 1, C]
        att = self.conv(att)                             # [B, 1, C]
        att = att.transpose(-1, -2).unsqueeze(-1)        # [B, C, 1, 1]
        return x * torch.sigmoid(att)


def create_attn_module(attn_type, dim, **kwargs):
    """Factory for attention modules. Returns nn.Identity() for 'none'."""
    if attn_type is None or attn_type.lower() in ('none', ''):
        return nn.Identity()
    if attn_type.lower() == 'ema':
        return EMA(dim, **kwargs)
    if attn_type.lower() == 'simam':
        return SimAM(**kwargs)
    if attn_type.lower() == 'eca':
        return ECA(dim, **kwargs)
    raise ValueError(f"Unknown attn_type: {attn_type}. Choose from: ema, simam, eca, none")

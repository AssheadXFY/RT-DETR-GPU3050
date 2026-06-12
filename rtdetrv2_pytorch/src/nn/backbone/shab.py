"""Spatial Hierarchical Attention Block (SHAB) -- from FSDETR/SHViT CVPR 2024.

SHAB replaces standard bottleneck blocks with single-head spatial self-attention,
capturing both local details (via depthwise conv) and global dependencies (via SHSA).
"""
import torch
import torch.nn as nn
from ...core import register


__all__ = ['SHAB', 'SHSABlock']


class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d with constant weight init for BN."""
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    """Residual wrapper: output = fn(x) + x."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SHSA(nn.Module):
    """Single-Head Self-Attention.

    Splits channels: pdim channels go through single-head attention,
    remaining channels are kept as identity bypass.
    """
    def __init__(self, dim, qk_dim=16, pdim=64):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.pre_norm = nn.GroupNorm(1, pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.SiLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        return self.proj(torch.cat([x1, x2], dim=1))


class SHSABlock_FFN(nn.Module):
    """Feed-forward network: pw1 -> SiLU -> pw2 (bn_weight_init=0)."""
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


@register()
class SHSABlock(nn.Module):
    """Single SHSABlock: DWConv -> SHSA -> FFN, each with residual."""
    def __init__(self, dim, qk_dim=16, pdim=64):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


@register()
class SHAB(nn.Module):
    """Spatial Hierarchical Attention Block -- a drop-in for backbone stage blocks.

    Usage: Replace a RepVggBlock or Bottleneck-based module with SHAB(c1, c2, n).
    Default pdim=min(64, c2//4) for compatibility with various channel sizes.
    """
    def __init__(self, c1, c2, num_blocks=3):
        super().__init__()
        hidden = c2
        self.conv_in = Conv2d_BN(c1, hidden, 1)
        pdim_default = min(64, hidden // 4)
        self.m = nn.ModuleList([SHSABlock(hidden, pdim=pdim_default) for _ in range(num_blocks)])
        self.conv_out = Conv2d_BN(hidden, c2, 1, bn_weight_init=0)

    def forward(self, x):
        x = self.conv_in(x)
        for m in self.m:
            x = m(x)
        return self.conv_out(x)

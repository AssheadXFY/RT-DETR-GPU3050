"""Enhanced Encoder Layer: ASSA + SEFN + Mona + DyT.

Adaptive Sparse Self-Attention (ASSA)  — CVPR 2024 (Adapt or Perish)
Spatial-Enhanced FFN (SEFN)           — WACV 2024 (SEMNet)
Multi-scale Adapter (Mona)            — CVPR 2025
Dynamic Tanh (DyT)                    — CVPR 2025 (Transformers without Norm)

All four modules are fully self-contained with native PyTorch only.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DynamicTanh — replaces LayerNorm with tanh(alpha * x)
# ---------------------------------------------------------------------------
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# LayerNorm2d (spatial-first, for Mona)
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# Mona — CVPR 2025 multi-scale depthwise conv adapter
# ---------------------------------------------------------------------------
class MonaOp(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, 5, padding=2, groups=ch)
        self.conv3 = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        out = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + x
        return out + self.proj(out)


class Mona(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim, 1, 1))
        self.down = nn.Conv2d(dim, 64, 1)
        self.adapter = MonaOp(64)
        self.up = nn.Conv2d(64, dim, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm(x) * self.gamma + x * self.gammax
        h = self.adapter(F.gelu(self.down(x)))
        return x + self.up(self.drop(h))


# ---------------------------------------------------------------------------
# SEFN — WACV 2024 Spatial-Enhanced FFN
# ---------------------------------------------------------------------------
class SEFN(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                 groups=hidden * 2, bias=False)
        self.fusion = nn.Conv2d(hidden + dim, hidden, 1, bias=False)
        self.dwconv_f = nn.Conv2d(hidden, hidden, 3, padding=1,
                                   groups=hidden, bias=False)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=False)
        # spatial branch
        self.pool = nn.AvgPool2d(2)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, spatial):
        proj = self.proj_in(x)
        x1, x2 = self.dwconv(proj).chunk(2, dim=1)
        # spatial context
        y = self.spatial_conv(self.pool(spatial))
        y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.dwconv_f(self.fusion(torch.cat([x1, y], dim=1)))
        return self.proj_out(F.gelu(x1) * x2)


# ---------------------------------------------------------------------------
# Adaptive Sparse Self-Attention — CVPR 2024
# (simplified: native PyTorch, no einops / timm)
# ---------------------------------------------------------------------------
def _window_partition(x, ws):
    """x: [B, H, W, C] -> [B*nW, ws, ws, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    """windows: [B*nW, ws, ws, C] -> [B, H, W, C]"""
    B = windows.shape[0] * ws * ws // (H * W)
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class WindowSparseAttention(nn.Module):
    """Window-based self-attention with softmax + ReLU^2 dual branches."""

    def __init__(self, dim, win_size, num_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        # learnable branch weight
        self.w = nn.Parameter(torch.ones(2))
        # relative position bias
        Wh, Ww = (win_size, win_size) if isinstance(win_size, int) else (win_size[0], win_size[1])
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        # precompute index
        ch = torch.arange(Wh)
        cw = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij'))
        coords_f = coords.reshape(2, -1)
        rc = coords_f[:, :, None] - coords_f[:, None, :]
        rc = rc.permute(1, 2, 0).contiguous()
        rc[:, :, 0] += Wh - 1
        rc[:, :, 1] += Ww - 1
        rc[:, :, 0] *= 2 * Ww - 1
        self.register_buffer('rel_idx', rc.sum(-1))

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        # relative bias
        bias = self.rel_bias[self.rel_idx.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1)
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            Nq = attn.shape[-1]  # attention target dim
            attn = attn.view(B // nW, nW, self.num_heads, N, Nq) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, Nq)
        # dual-branch (softmax + ReLU^2)
        a0 = attn.softmax(dim=-1)
        a1 = F.relu(attn) ** 2
        w1 = self.w[0].exp() / self.w.exp().sum()
        w2 = self.w[1].exp() / self.w.exp().sum()
        attn = a0 * w1 + a1 * w2
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(self.drop(x))


class AdaptiveSparseSA(nn.Module):
    """2D adaptive sparse self-attention block (windowed, shifted)."""

    def __init__(self, dim, num_heads=8, win_size=4, shift_size=2, mlp_ratio=4.,
                 dropout=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowSparseAttention(dim, win_size, num_heads, attn_drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        # light FFN (FRFN-style, partial conv + gated MLP)
        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv
        self.pconv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_seq = x.flatten(2).transpose(1, 2)  # [B, L, C]
        shortcut = x_seq
        # ---- attention ----
        z = self.norm1(x_seq)
        z = z.view(B, H, W, C)
        if self.shift_size > 0:
            z = torch.roll(z, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # partition
        z_win = _window_partition(z, self.win_size)  # [B*nW, ws, ws, C]
        nW = z_win.shape[0]
        z_win = z_win.view(nW, self.win_size * self.win_size, C)
        # shift mask
        attn_mask = None
        if self.shift_size > 0:
            sm = torch.zeros((1, H, W, 1), device=x.device)
            ws = self.win_size
            ss = self.shift_size
            slices = [(slice(0, -ws), slice(0, -ws)),
                      (slice(0, -ws), slice(-ws, -ss)),
                      (slice(0, -ws), slice(-ss, None)),
                      (slice(-ws, -ss), slice(0, -ws)),
                      (slice(-ws, -ss), slice(-ws, -ss)),
                      (slice(-ws, -ss), slice(-ss, None)),
                      (slice(-ss, None), slice(0, -ws)),
                      (slice(-ss, None), slice(-ws, -ss)),
                      (slice(-ss, None), slice(-ss, None))]
            for i, (hs, ws_) in enumerate(slices):
                sm[:, hs, ws_, :] = i
            sm_win = _window_partition(sm, ws).view(-1, ws * ws)
            attn_mask = sm_win[:, None, :] - sm_win[:, :, None]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(
                attn_mask == 0, 0.0)
        z_attn = self.attn(z_win, attn_mask)
        z_attn = z_attn.view(nW, ws, ws, C)
        z = _window_reverse(z_attn, ws, H, W)
        if self.shift_size > 0:
            z = torch.roll(z, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        z = z.view(B, L, C)
        x_seq = shortcut + self.drop_path(z)
        # ---- FFN ----
        shortcut2 = x_seq
        z2 = self.norm2(x_seq)
        # partial conv (FRFN-style)
        z2_2d = z2.transpose(1, 2).view(B, C, H, W)
        c1, cu = torch.split(z2_2d, [self.dim_conv, self.dim_untouched], dim=1)
        c1 = self.pconv(c1)
        z2_2d = torch.cat([c1, cu], dim=1)
        z2_2d = z2_2d.view(B, C, L).transpose(1, 2)
        # gated MLP
        gated = self.fc1(z2_2d)
        g1, g2 = gated.chunk(2, dim=-1)
        g1_2d = g1.transpose(1, 2).reshape(B, -1, H, W)
        g1_2d = self.dwconv(g1_2d)
        g1 = g1_2d.flatten(2).transpose(1, 2)
        z2 = self.act(g1) * g2
        z2 = self.fc2(z2)
        x_seq = shortcut2 + self.drop_path(z2)
        return x_seq  # [B, L, C]


# ---------------------------------------------------------------------------
# Composite Enhanced Encoder Layer
# ---------------------------------------------------------------------------
class EnhancedEncoderLayer(nn.Module):
    """ASSA + SEFN + Mona + DyT composite encoder layer.

    Replaces the standard TransformerEncoderLayer in the HybridEncoder.
    Input/Output: [B, C, H, W] (2D spatial format).
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.,
                 activation='gelu', normalize_before=False):
        super().__init__()
        del dim_feedforward, activation, normalize_before  # handled internally
        self.assa = AdaptiveSparseSA(d_model, num_heads=nhead, win_size=4, shift_size=2)

        self.norm1 = DynamicTanh(d_model, channels_last=False)
        self.norm2 = DynamicTanh(d_model, channels_last=False)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.sefn = SEFN(d_model, expansion=2.0)
        self.mona1 = Mona(d_model)
        self.mona2 = Mona(d_model)

    def forward(self, src, src_mask=None, pos=None):
        # src: [B, C, H, W]
        src_spatial = src
        # ASSA attention
        attn_out = self.assa(src_spatial)  # [B, L, C]
        B, L, C = attn_out.shape
        H = W = int(L ** 0.5)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        src_spatial = src_spatial + self.drop1(attn_out)
        src_spatial = self.norm1(src_spatial)
        src_spatial = self.mona1(src_spatial)
        # SEFN
        ffn_out = self.sefn(src_spatial, src)
        src_spatial = src_spatial + self.drop2(ffn_out)
        src_spatial = self.mona2(self.norm2(src_spatial))
        return src_spatial

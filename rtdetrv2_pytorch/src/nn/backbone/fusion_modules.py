"""Cross-layer Modulation and Frequency-Spatial Fusion modules.
FCM modules from FCM-main (Feature Cross-layer Modulation).
FreqSpatial from FSDETR (Frequency-Spatial Feature Enhancement).
"""
import torch
import torch.nn as nn
import numpy as np
from ...core import register


__all__ = ['Channel', 'Spatial', 'FCM', 'FCM_1', 'FCMBlock', 'ScharrConv', 'FreqSpatial', 'FreqSpatialBlock']


# ==================== FCM Family (from FCM-main) ====================

class Channel(nn.Module):
    """Channel attention via depthwise conv -> global avg pool -> sigmoid."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)
        return x6


class Spatial(nn.Module):
    """Spatial attention via Conv1x1 -> BN -> sigmoid."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)
        return x6


class _ConvBN(nn.Module):
    """Conv2d + BatchNorm2d + optional activation."""
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


@register()
class FCM(nn.Module):
    """Feature Cross-layer Modulation (1:3 split, with final 1x1 fusion).

    Splits channels into 1:3 groups: the small group gets spatial refinement
    (3x3 convs), the large group gets semantic transformation (1x1 conv), then
    both branches cross-modulate each other via spatial/channel attention.
    """
    def __init__(self, c1, c2):
        super().__init__()
        dim = c1
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = _ConvBN(dim // 4, dim // 4, 3)
        self.conv12 = _ConvBN(dim // 4, dim // 4, 3)
        self.conv123 = _ConvBN(dim // 4, dim, 1)
        self.conv2 = _ConvBN(dim - dim // 4, dim, 1)
        self.conv3 = _ConvBN(dim, c2, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        out = self.spatial(x4) * x3 + self.channel(x3) * x4
        return self.conv3(out)


@register()
class FCM_1(nn.Module):
    """FCM variant without final 1x1. Lighter, same 1:3 split."""
    def __init__(self, c1, c2):
        super().__init__()
        dim = c1
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = _ConvBN(dim // 4, dim // 4, 3)
        self.conv12 = _ConvBN(dim // 4, dim // 4, 3)
        self.conv123 = _ConvBN(dim // 4, c2, 1)
        self.conv2 = _ConvBN(dim - dim // 4, c2, 1)
        self.spatial = Spatial(c2)
        self.channel = Channel(c2)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        out = self.spatial(x4) * x3 + self.channel(x3) * x4
        return out


@register()
class FCMBlock(nn.Module):
    """CSP-like block using FCM as the bottleneck.

    Takes concatenated features (channel=dim*2) from FPN/PAN, processes
    though FCM bottleneck blocks, fuses via split + fuse.
    """
    def __init__(self, c1, c2, num_blocks=3):
        super().__init__()
        hidden = c2
        self.conv_a = _ConvBN(c1, hidden, 3, act=True)
        self.conv_b = _ConvBN(c1, hidden, 1, act=True)
        self.m = nn.ModuleList([FCM_1(hidden, hidden) for _ in range(num_blocks)])
        self.conv_out = _ConvBN(hidden, c2, 1, act=True)

    def forward(self, x):
        y = [self.conv_a(x)]
        for m in self.m:
            y.append(m(y[-1]))
        y = torch.concat(y, dim=1)
        return self.conv_out(y)


# ==================== FreqSpatial Family (from FSDETR) ====================

class ScharrConv(nn.Module):
    """Fixed Scharr edge-detection convolution (no learnable params).
    Extracts edge magnitude from spatial gradients.
    """
    def __init__(self, channel):
        super().__init__()
        kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
        kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)
        kx = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.scharr_x = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.scharr_y = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.scharr_x.weight.data = kx.expand(channel, 1, 3, 3).clone()
        self.scharr_y.weight.data = ky.expand(channel, 1, 3, 3).clone()
        self.scharr_x.requires_grad = False
        self.scharr_y.requires_grad = False

    def forward(self, x):
        return self.scharr_x(x) * 0.5 + self.scharr_y(x) * 0.5


class FreqSpatial(nn.Module):
    """Dual-branch frequency-spatial feature processor.

    Spatial branch: Scharr edge detection -> spatial convs.
    Frequency branch: FFT -> conv in freq domain -> iFFT.
    Fuses both via addition.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sed = ScharrConv(in_channels)
        self.spatial_conv1 = _ConvBN(in_channels, in_channels)
        self.spatial_conv2 = _ConvBN(in_channels, in_channels)
        self.fft_conv = _ConvBN(in_channels * 2, in_channels * 2)
        self.fft_conv2 = _ConvBN(in_channels, in_channels)
        self.final_conv = _ConvBN(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # spatial path
        sf = self.sed(x)
        sf = self.spatial_conv1(sf)
        sf = self.spatial_conv2(sf + x)

        # frequency path
        fft = torch.fft.rfft2(x, norm='ortho')
        real = torch.unsqueeze(torch.real(fft), dim=-1)
        imag = torch.unsqueeze(torch.imag(fft), dim=-1)
        fft_cat = torch.cat((real, imag), dim=-1)
        fft_cat = fft_cat.permute(0, 1, 4, 2, 3).reshape(B, C * 2, fft.size(-2), fft.size(-1))
        fft_cat = self.fft_conv(fft_cat)
        fft_cat = fft_cat.reshape(B, C, 2, fft.size(-2), fft.size(-1))
        fft_cat = fft_cat.permute(0, 1, 3, 4, 2).contiguous()
        fft_cat = torch.view_as_complex(fft_cat)
        ff = torch.fft.irfft2(fft_cat, s=(H, W), norm='ortho')
        ff = self.fft_conv2(ff)

        return self.final_conv(sf + ff)


@register()
class FreqSpatialBlock(nn.Module):
    """CSP-like block using FreqSpatial as the bottleneck.
    Replaces CSPRepLayer in FPN/PAN neck with frequency-spatial awareness.
    """
    def __init__(self, c1, c2, num_blocks=3):
        super().__init__()
        hidden = c2
        self.conv1 = _ConvBN(c1, hidden, 1)
        self.conv2 = _ConvBN(c1, hidden, 1)
        self.m = nn.ModuleList([FreqSpatial(hidden) for _ in range(num_blocks)])
        self.conv3 = _ConvBN(hidden, c2, 1) if hidden != c2 else nn.Identity()

    def forward(self, x):
        x_a = self.conv1(x)
        for m in self.m:
            x_a = m(x_a)
        x_b = self.conv2(x)
        return self.conv3(x_a + x_b)

"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from ..core import register


__all__ = ['GradScaler']


class _DeviceGradScaler:
    """GradScaler wrapper that auto-detects device type (CUDA/NPU)."""
    def __new__(cls, *args, **kwargs):
        from ..misc.dist_utils import device_type
        dt = device_type()
        if dt == 'cpu':
            dt = 'cuda'
        return torch.amp.GradScaler(device_type=dt, *args, **kwargs)


GradScaler = register()(_DeviceGradScaler)

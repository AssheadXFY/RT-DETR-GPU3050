"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from ..core import register


__all__ = ['GradScaler']


def _get_grad_scaler():
    from ..misc.dist_utils import device_module, device_type
    dm = device_module()
    if dm is not None and hasattr(dm, 'amp') and hasattr(dm.amp, 'GradScaler'):
        return dm.amp.GradScaler
    return torch.amp.GradScaler  # PyTorch >= 2.0


@register()
class GradScaler:
    """GradScaler wrapper that auto-detects device type (CUDA/NPU).
    Supports both new torch.amp API (PT>=2.0) and old torch.npu.amp API.
    """
    def __new__(cls, *args, **kwargs):
        from ..misc.dist_utils import device_type
        dt = device_type()
        if dt == 'cpu':
            dt = 'cuda'
        Scaler = _get_grad_scaler()
        try:
            return Scaler(device_type=dt, *args, **kwargs)
        except TypeError:
            return Scaler(*args, **kwargs)

"""FLOPs / FPS / Params / GPU Memory benchmark for RT-DETR variants.

Usage:
    python tools/benchmark_efficiency.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml
    python tools/benchmark_efficiency.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_fcm.yml
    python tools/benchmark_efficiency.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml

Requirements: pip install thop (optional, fallback to torch.profiler)
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(num: float, unit: str = '', precision: int = 2) -> str:
    return f'{num:.{precision}f}{unit}'


def clean_table(header: list[str], rows: list[list[str]]) -> str:
    """Format a simple Markdown table."""
    col_w = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))
    sep = '| ' + ' | '.join('-' * w for w in col_w) + ' |'
    hdr = '| ' + ' | '.join(h.ljust(w) for h, w in zip(header, col_w)) + ' |'
    body = '\n'.join(
        '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(row, col_w)) + ' |'
        for row in rows
    )
    return f'{hdr}\n{sep}\n{body}'


# ---------------------------------------------------------------------------
# FLOPs via thop (preferred) or torch.profiler (fallback)
# ---------------------------------------------------------------------------

def measure_flops_thop(model: nn.Module, device: torch.device,
                       shape: list[int]) -> float:
    """Return GFLOPs using thop. Falls back on error."""
    from thop import profile as thop_profile
    x = torch.randn(*shape, device=device)
    # FreqSpatial uses FFT; thop may trace F.interpolate with size= keyword
    total_ops, _ = thop_profile(model, inputs=(x,), verbose=False)
    return total_ops / 1e9


def measure_flops_torchprofiler(model: nn.Module, device: torch.device,
                                shape: list[int]) -> float:
    """Return GFLOPs using torch.profiler (fallback)."""
    x = torch.randn(*shape, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
    ) as p:
        _ = model(x)

    total_flops = sum(e.flops for e in p.key_averages() if e.flops is not None and e.flops > 0)
    return total_flops / 1e9


def measure_flops(model: nn.Module, device: torch.device,
                  shape: list[int] = [1, 3, 640, 640]) -> OrderedDict:
    """Measure FLOPs, trying thop first then torch.profiler."""
    model.eval()
    results = OrderedDict()

    # --- thop ---
    try:
        flops = measure_flops_thop(model, device, shape)
        results['GFLOPs (thop)'] = fmt(flops)
    except Exception as e:
        results['GFLOPs (thop)'] = f'ERR: {str(e)[:40]}'

    # --- torch.profiler ---
    try:
        flops2 = measure_flops_torchprofiler(model, device, shape)
        results['GFLOPs (profiler)'] = fmt(flops2)
    except Exception as e:
        results['GFLOPs (profiler)'] = f'ERR: {str(e)[:40]}'

    return results


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

def measure_params(model: nn.Module) -> OrderedDict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    r = OrderedDict()
    r['Params (M)'] = fmt(total / 1e6, precision=2)
    r['Trainable (M)'] = fmt(trainable / 1e6, precision=2)
    return r


# ---------------------------------------------------------------------------
# FPS / Latency
# ---------------------------------------------------------------------------

def measure_fps(model: nn.Module, device: torch.device,
                shape: list[int] = [1, 3, 640, 640],
                warmup: int = 50, iters: int = 300) -> OrderedDict:
    """Measure pure-inference FPS (eval mode, no AMP, no grad)."""
    model.eval()
    x = torch.randn(*shape, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # timed loop
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    r = OrderedDict()
    r['FPS (bs=1)'] = fmt(iters / elapsed, precision=1)
    r['Latency (ms)'] = fmt(elapsed / iters * 1000, precision=2)
    return r


# ---------------------------------------------------------------------------
# GPU Memory
# ---------------------------------------------------------------------------

def measure_gpu_memory(model: nn.Module, device: torch.device,
                       shape: list[int] = [1, 3, 640, 640]) -> OrderedDict:
    if device.type != 'cuda':
        return OrderedDict([('Peak Mem', 'N/A')])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    x = torch.randn(*shape, device=device)
    with torch.no_grad():
        _ = model(x)

    r = OrderedDict()
    r['Peak Mem (GB)'] = fmt(torch.cuda.max_memory_allocated(device) / 1e9, precision=2)
    return r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def benchmark_one(cfg_path: str, device_str: str, input_size: int) -> OrderedDict:
    """Run all benchmarks for a single config. Returns a flat dict."""
    device = torch.device(device_str)

    # build model from config (random weights)
    cfg = YAMLConfig(cfg_path, device=device_str)
    model: nn.Module = cfg.model
    model = model.to(device)

    name = os.path.basename(cfg_path).replace('rtdetrv2_r18vd_120e_coco', '') \
                                         .replace('.yml', '') \
                                         .strip('_') or 'Baseline'
    shape = [1, 3, input_size, input_size]

    results = OrderedDict()
    results['Model'] = name

    # params
    results.update(measure_params(model))

    # flops
    print(f'  [{name}] measuring FLOPs ...')
    results.update(measure_flops(model, device, shape))

    # fps
    print(f'  [{name}] measuring FPS (warmup=50, iters=300) ...')
    results.update(measure_fps(model, device, shape))

    # gpu memory
    results.update(measure_gpu_memory(model, device, shape))

    return results


def main():
    parser = argparse.ArgumentParser(description='RT-DETR Efficiency Benchmark')
    parser.add_argument('-c', '--configs', nargs='+', required=True,
                        help='One or more YAML config paths')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-s', '--input-size', type=int, default=640,
                        help='Square input size (default 640)')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--csv', action='store_true', help='Also output CSV rows')
    args = parser.parse_args()

    # Set default FPS params as globals (used by measure_fps via closure, or pass through)
    # We rebuild measure_fps to use args.warmup and args.iters — just pass them.
    all_results = []
    for cfg in args.configs:
        print(f'\n{"="*60}')
        print(f'Benchmarking: {cfg}')
        print(f'{"="*60}')
        results = benchmark_one(cfg, args.device, args.input_size)
        all_results.append(results)

    # Print table
    print('\n')
    if all_results:
        header = list(all_results[0].keys())
        rows = [[r.get(k, '') for k in header] for r in all_results]
        print(clean_table(header, rows))

    # CSV
    if args.csv:
        print('\n# --- CSV ---')
        print(','.join(header))
        for row in rows:
            print(','.join(row))


if __name__ == '__main__':
    main()

"""Detection visualization for RT-DETR model variants.

CVPR/ECCV paper standard: matplotlib-based, publication-quality.
Produces clean side-by-side comparison figures.

Usage:
    python tools/visualize_compare.py \
        -c1 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r1 ckpt/baseline.pth --name1 Baseline \
        -c2 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml -r2 ckpt/freq.pth --name2 FreqSpatial \
        --auto --sample-size 500 --top-k 10
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


# ---- paper-ready matplotlib style ----
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

# Distinct class colors (CVPR palette)
CLASS_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
CLASS_COLORS = np.vstack([CLASS_COLORS, plt.cm.tab20b(np.linspace(0, 1, 20))])
CLASS_COLORS = np.vstack([CLASS_COLORS, plt.cm.tab20c(np.linspace(0, 1, 20))])
CLASS_COLORS = np.vstack([CLASS_COLORS, plt.cm.Set3(np.linspace(0, 1, 12))])
CLASS_COLORS = np.vstack([CLASS_COLORS, plt.cm.Paired(np.linspace(0, 1, 12))])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(state, strict=False)

    class M(nn.Module):
        def __init__(self, model, pp):
            super().__init__()
            self.model = model.deploy()
            self.pp = pp.deploy()
        def forward(self, x, s):
            o = self.model(x)
            return self.pp(o, s)
    return M(cfg.model, cfg.postprocessor).to(device).eval()


@torch.no_grad()
def detect(model, image, device, input_size=640, score_thr=0.4):
    w, h = image.size
    orig = torch.tensor([w, h])[None].to(device)
    x = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])(image)[None].to(device)
    labels, boxes, scores = model(x, orig)
    keep = scores[0] >= score_thr
    return (boxes[0][keep].cpu().numpy(),
            labels[0][keep].cpu().numpy().astype(int),
            scores[0][keep].cpu().numpy())


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-8)


def match_preds(boxes, labels, scores, gt_anns, iou_thr=0.5):
    n_p, n_g = len(boxes), len(gt_anns)
    is_tp = np.zeros(n_p, dtype=bool)
    if n_p == 0:
        return is_tp, np.zeros(n_g, dtype=bool)
    if n_g == 0:
        return is_tp, np.zeros(0, dtype=bool)
    gt_arr = np.array([[a['bbox'][0], a['bbox'][1],
                        a['bbox'][0] + a['bbox'][2],
                        a['bbox'][1] + a['bbox'][3],
                        a['category_id'] - 1] for a in gt_anns])
    order = np.argsort(-scores)
    gt_m = np.zeros(n_g, dtype=bool)
    for i in order:
        cls = labels[i]
        cand = [j for j in range(n_g) if not gt_m[j] and int(gt_arr[j, 4]) == cls]
        if not cand:
            continue
        ious = box_iou(boxes[i], gt_arr[cand])
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr:
            is_tp[i] = True
            gt_m[cand[best]] = True
    return is_tp, gt_m


# ---------------------------------------------------------------------------
# paper-standard matplotlib visualization
# ---------------------------------------------------------------------------

def _cls_color(cls_id):
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


def draw_boxes_ax(ax, image_np, boxes, labels, scores, is_tp=None,
                  draw_fp=False, gain_mask=None, score_thr=0.0):
    """Draw detection boxes on a matplotlib axis in paper-standard style."""
    ax.imshow(image_np)
    ax.axis('off')

    h, w = image_np.shape[:2]

    for i in range(len(boxes)):
        if scores[i] < score_thr:
            continue

        x1, y1, x2, y2 = boxes[i]
        cls_id = int(labels[i])

        # Determine style
        if gain_mask is not None and i < len(gain_mask) and gain_mask[i]:
            # Gain: detected by FreqSpatial, missed by Baseline
            color = '#00CC00'
            lw = 2.5
            style = '-'
            alpha = 1.0
        elif is_tp is not None:
            if is_tp[i]:
                color = _cls_color(cls_id)  # color per class
                lw = 1.5
                style = '-'
                alpha = 0.9
            else:
                if not draw_fp:
                    continue
                color = 'red'
                lw = 1.0
                style = '--'
                alpha = 0.5
        else:
            color = _cls_color(cls_id)
            lw = 1.5
            style = '-'
            alpha = 0.9

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=lw, edgecolor=color, facecolor='none',
            linestyle=style, alpha=alpha)
        ax.add_patch(rect)

        # Scale font based on box size relative to image
        box_size_ratio = max(x2 - x1, y2 - y1) / max(w, h)
        fontsize = max(4, min(10, 10 * box_size_ratio * 5))

        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'{cls_id}'
        label_txt = f'{name}'  # class name only, no score (cleaner for paper)

        # Position text above the box
        ax.text(x1, max(0, y1 - 2), label_txt,
                fontsize=fontsize, color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                          edgecolor='none', alpha=alpha),
                verticalalignment='bottom')


def draw_gt_ax(ax, image_np, gt_anns):
    """Draw GT boxes on a matplotlib axis."""
    ax.imshow(image_np)
    ax.axis('off')

    for ann in gt_anns:
        x, y, w, h = ann['bbox']
        cls_id = ann['category_id'] - 1
        color = _cls_color(cls_id)
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5, edgecolor=color, facecolor='none',
            linestyle='-', alpha=0.7)
        ax.add_patch(rect)
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'{cls_id}'
        ax.text(x, max(0, y - 2), name,
                fontsize=6, color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                          edgecolor='none', alpha=0.7),
                verticalalignment='bottom')


def build_comparison(image, gt_anns,
                     boxes_a, labels_a, scores_a, is_tp_a, name_a,
                     boxes_b, labels_b, scores_b, is_tp_b, gt_m_a, name_b,
                     score_thr=0.4):
    """
    3-panel CVPR-standard figure:
      (a) Ground Truth
      (b) Model A predictions
      (c) Model B predictions with gains highlighted
    """
    img_np = np.array(image)

    # Filter by threshold
    keep_a = scores_a >= score_thr
    keep_b = scores_b >= score_thr

    boxes_a_k = boxes_a[keep_a]; labels_a_k = labels_a[keep_a]
    scores_a_k = scores_a[keep_a]; is_tp_a_k = is_tp_a[keep_a]

    boxes_b_k = boxes_b[keep_b]; labels_b_k = labels_b[keep_b]
    scores_b_k = scores_b[keep_b]; is_tp_b_k = is_tp_b[keep_b]

    # Compute gains mask
    gt_arr = np.array([[a['bbox'][0], a['bbox'][1],
                        a['bbox'][0] + a['bbox'][2],
                        a['bbox'][1] + a['bbox'][3],
                        a['category_id'] - 1] for a in gt_anns])
    gain_mask = np.zeros(len(boxes_b_k), dtype=bool)
    gains = 0
    for i in range(len(boxes_b_k)):
        if is_tp_b_k[i]:
            cls = labels_b_k[i]
            cand = [j for j in range(len(gt_arr)) if int(gt_arr[j, 4]) == cls]
            if cand:
                ious = box_iou(boxes_b_k[i], gt_arr[cand])
                best_l = cand[int(np.argmax(ious))]
                if ious.max() >= 0.5 and best_l < len(gt_m_a) and not gt_m_a[best_l]:
                    gain_mask[i] = True
                    gains += 1

    tp_a = int(is_tp_a_k.sum()) if len(is_tp_a_k) > 0 else 0
    fp_a = len(boxes_a_k) - tp_a
    tp_b = int(is_tp_b_k.sum()) if len(is_tp_b_k) > 0 else 0
    fp_b = len(boxes_b_k) - tp_b
    n_gt = len(gt_anns)

    # ---- Build figure ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor='white')
    plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.92, bottom=0.01)

    # (a) Ground Truth
    draw_gt_ax(axes[0], img_np, gt_anns)
    axes[0].set_title(f'(a) Ground Truth ({n_gt} objects)', fontweight='bold', pad=4)

    # (b) Baseline
    draw_boxes_ax(axes[1], img_np, boxes_a_k, labels_a_k, scores_a_k,
                  is_tp=is_tp_a_k, draw_fp=True, score_thr=0)
    axes[1].set_title(f'(b) {name_a}\nTP={tp_a}/{n_gt}  FP={fp_a}',
                      fontweight='bold', pad=4)

    # (c) FreqSpatial with gains
    draw_boxes_ax(axes[2], img_np, boxes_b_k, labels_b_k, scores_b_k,
                  is_tp=is_tp_b_k, draw_fp=True, gain_mask=gain_mask, score_thr=0)
    axes[2].set_title(f'(c) {name_b}\nTP={tp_b}/{n_gt}  FP={fp_b}  Gains={gains}',
                      fontweight='bold', pad=4)

    # Overall title
    fig.suptitle(f'{name_b} vs {name_a} — Detection Comparison',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.close(fig)
    return fig, {'tp_a': tp_a, 'fp_a': fp_a, 'tp_b': tp_b, 'fp_b': fp_b,
                 'gains': gains, 'n_gt': n_gt, 'diff': tp_b - tp_a}


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

def load_coco(ann_file):
    with open(ann_file) as f:
        coco = json.load(f)
    by_img = defaultdict(list)
    for a in coco['annotations']:
        by_img[a['image_id']].append(a)
    return by_img, {i['id']: i for i in coco['images']}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Detection Comparison')
    parser.add_argument('-c1', '--config1', required=True)
    parser.add_argument('-r1', '--resume1', required=True)
    parser.add_argument('--name1', default='Baseline')
    parser.add_argument('-c2', '--config2', required=True)
    parser.add_argument('-r2', '--resume2', required=True)
    parser.add_argument('--name2', default='FreqSpatial')
    parser.add_argument('--im-dir', default='dataset/coco/val2017')
    parser.add_argument('--ann-file', default='dataset/coco/annotations/instances_val2017.json')
    parser.add_argument('--out-dir', default='output/compare')
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-images', type=int, default=30)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f'Loading {args.name1}: {args.resume1}')
    m_a = load_model(args.config1, args.resume1, args.device)
    print(f'Loading {args.name2}: {args.resume2}')
    m_b = load_model(args.config2, args.resume2, args.device)
    print('Loading COCO ...')
    anns_by_img, imgs_info = load_coco(args.ann_file)

    scan_n = min(args.sample_size if args.auto else args.num_images, len(imgs_info))
    ids = random.sample(list(imgs_info.keys()), scan_n)

    print(f'Scanning {scan_n} images ...')
    scored = []
    for idx, img_id in enumerate(ids):
        info = imgs_info[img_id]
        image = Image.open(os.path.join(args.im_dir, info['file_name'])).convert('RGB')
        gt = anns_by_img.get(img_id, [])
        if not gt:
            continue

        ba, la, sa = detect(m_a, image, args.device, score_thr=args.score_thr)
        bb, lb, sb = detect(m_b, image, args.device, score_thr=args.score_thr)
        is_tp_a, gt_m_a = match_preds(ba, la, sa, gt)
        is_tp_b, gt_m_b = match_preds(bb, lb, sb, gt)
        tp_a, tp_b = int(is_tp_a.sum()), int(is_tp_b.sum())

        gt_arr = np.array([[a['bbox'][0], a['bbox'][1],
                            a['bbox'][0] + a['bbox'][2],
                            a['bbox'][1] + a['bbox'][3],
                            a['category_id'] - 1] for a in gt])
        gains = 0
        for i in range(len(bb)):
            if is_tp_b[i]:
                cls = lb[i]
                cand = [j for j in range(len(gt_arr)) if int(gt_arr[j, 4]) == cls]
                if cand:
                    ious = box_iou(bb[i], gt_arr[cand])
                    best_l = cand[int(np.argmax(ious))]
                    if ious.max() >= 0.5 and best_l < len(gt_m_a) and not gt_m_a[best_l]:
                        gains += 1

        scored.append({
            'img_id': img_id, 'info': info,
            'boxes_a': ba, 'labels_a': la, 'scores_a': sa, 'is_tp_a': is_tp_a,
            'boxes_b': bb, 'labels_b': lb, 'scores_b': sb, 'is_tp_b': is_tp_b,
            'gt_m_a': gt_m_a, 'gt': gt,
            'tp_a': tp_a, 'tp_b': tp_b, 'n_gt': len(gt),
            'gains': gains, 'diff': tp_b - tp_a,
        })
        if (idx + 1) % 50 == 0:
            print(f'  [{idx+1}/{scan_n}] ...')

    scored.sort(key=lambda x: (x['gains'], x['diff']), reverse=True)
    top_n = min(args.top_k if args.auto else len(scored), len(scored))

    if args.auto:
        diffs = [s['diff'] for s in scored]
        gains_list = [s['gains'] for s in scored]
        print(f'\nDiff: max={max(diffs)} min={min(diffs)} mean={np.mean(diffs):.2f}')
        print(f'Gains: max={max(gains_list)} mean={np.mean(gains_list):.2f}')
        print(f'{args.name2} wins: {sum(1 for d in diffs if d > 0)}/{len(scored)}')

    print(f'\nRendering top {top_n} ...')
    for rank, s in enumerate(scored[:top_n]):
        image = Image.open(os.path.join(args.im_dir, s['info']['file_name'])).convert('RGB')
        fig, stats = build_comparison(
            image, s['gt'],
            s['boxes_a'], s['labels_a'], s['scores_a'], s['is_tp_a'], args.name1,
            s['boxes_b'], s['labels_b'], s['scores_b'], s['is_tp_b'],
            s['gt_m_a'], args.name2, score_thr=args.score_thr)

        fname = (f'rank{rank+1:02d}_{s["img_id"]:012d}_'
                 f'dTP{stats["diff"]:+d}_g{stats["gains"]}.pdf')
        fig.savefig(os.path.join(args.out_dir, fname))
        # Also save PNG for quick preview
        fname_png = fname.replace('.pdf', '.png')
        fig.savefig(os.path.join(args.out_dir, fname_png))
        plt.close(fig)

        print(f'  [{rank+1:2d}] {s["info"]["file_name"]}  '
              f'{args.name1}_TP={stats["tp_a"]}/{stats["n_gt"]}  '
              f'{args.name2}_TP={stats["tp_b"]}/{stats["n_gt"]}  '
              f'ΔTP={stats["diff"]:+d}  gains={stats["gains"]}  '
              f'→ {fname_png}')

    print(f'\nDone → {args.out_dir}/')


if __name__ == '__main__':
    main()

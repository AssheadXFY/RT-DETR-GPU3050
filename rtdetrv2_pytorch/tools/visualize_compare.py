"""Detection visualization for RT-DETR model variants.

Produces clean side-by-side comparison images for paper.

Usage:
    python tools/visualize_compare.py \
        -c1 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r1 ckpt/baseline.pth --name1 Baseline \
        -c2 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml -r2 ckpt/freq.pth --name2 FreqSpatial \
        --auto --sample-size 500 --top-k 15
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


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
    """Returns is_tp[n_preds], gt_matched[n_gt]."""
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
# drawing
# ---------------------------------------------------------------------------

FONT_FILE = None  # will be resolved


def _font(size):
    global FONT_FILE
    try:
        if FONT_FILE is None:
            FONT_FILE = ImageFont.truetype('arial.ttf', size)
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _class_name(cls_id):
    return COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'


def draw_comparison(image, gt_anns,
                    boxes_a, labels_a, scores_a, is_tp_a, name_a,
                    boxes_b, labels_b, scores_b, is_tp_b, gt_m_a, name_b):
    """
    Side-by-side: left = model A (blue), right = model B (red).
    GT boxes drawn faintly on both sides.
    Gains (B-TP, A-missed) drawn as thick green boxes on the B side.
    TP/FP counts in title bar.
    """
    im_a = image.copy()
    im_b = image.copy()
    dr_a = ImageDraw.Draw(im_a)
    dr_b = ImageDraw.Draw(im_b)
    font = _font(11)
    font_big = _font(13)

    # --- Draw GT on both sides (thin yellow) ---
    for ann in gt_anns:
        x, y, wb, hb = ann['bbox']
        x1, y1, x2, y2 = x, y, x + wb, y + hb
        for img, dr in [(im_a, dr_a), (im_b, dr_b)]:
            dr.rectangle([x1, y1, x2, y2], outline='#BBA800', width=1)

    # --- Model A: blue predictions ---
    tp_a, fp_a = 0, 0
    for i in range(len(boxes_a)):
        x1, y1, x2, y2 = boxes_a[i]
        txt = f'{_class_name(labels_a[i])} {scores_a[i]:.2f}'
        if is_tp_a[i]:
            tp_a += 1
            color = '#3399FF'
            width = 3
        else:
            fp_a += 1
            color = '#FF6666'
            width = 2
        dr_a.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tb = dr_a.textbbox((x1, max(0, y1 - 14)), txt, font=font)
        dr_a.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        dr_a.text((x1, max(0, y1 - 14)), txt if is_tp_a[i] else txt[:6],
                  fill='#FFFFFF', font=font)

    # --- Model B: red predictions ---
    tp_b, fp_b = 0, 0
    for i in range(len(boxes_b)):
        x1, y1, x2, y2 = boxes_b[i]
        txt = f'{_class_name(labels_b[i])} {scores_b[i]:.2f}'
        if is_tp_b[i]:
            tp_b += 1
            color = '#FF6644'
            width = 3
        else:
            fp_b += 1
            color = '#FF6666'
            width = 2
        dr_b.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tb = dr_b.textbbox((x1, max(0, y1 - 14)), txt, font=font)
        dr_b.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        dr_b.text((x1, max(0, y1 - 14)), txt if is_tp_b[i] else txt[:6],
                  fill='#FFFFFF', font=font)

    # --- Gains on B side: thick bright green rectangles ---
    gains = 0
    # For each TP in B, check if the matching GT was missed in A
    gt_arr = np.array([[a['bbox'][0], a['bbox'][1],
                        a['bbox'][0] + a['bbox'][2],
                        a['bbox'][1] + a['bbox'][3],
                        a['category_id'] - 1] for a in gt_anns])
    for i in range(len(boxes_b)):
        if is_tp_b[i]:
            cls = labels_b[i]
            cand = [j for j in range(len(gt_arr)) if int(gt_arr[j, 4]) == cls]
            if cand:
                ious = box_iou(boxes_b[i], gt_arr[cand])
                best_local = cand[int(np.argmax(ious))]
                if ious.max() >= 0.5 and best_local < len(gt_m_a) and not gt_m_a[best_local]:
                    gains += 1
                    x1, y1, x2, y2 = boxes_b[i]
                    # thick bright-green border to stand out
                    dr_b.rectangle([x1-2, y1-2, x2+2, y2+2],
                                   outline='#00FF00', width=5)
                    dr_b.rectangle([x1-4, y1-4, x2+4, y2+4],
                                   outline='#00FF00', width=2)

    # --- Assemble composite ---
    w = image.width
    gap = 6
    bar_h = 30
    total_w = w * 2 + gap
    total_h = bar_h + max(image.height, image.height)

    canvas = Image.new('RGB', (total_w, total_h), '#FAFAFA')
    dr_c = ImageDraw.Draw(canvas)

    # Title bar
    dr_c.rectangle([0, 0, total_w, bar_h], fill='#2a2a2a')
    n_gt = len(gt_anns)
    dr_c.text((8, 6), f'{name_a}  TP={tp_a}  FP={fp_a}  (GT={n_gt})',
              fill='#3399FF', font=font_big)
    dr_c.text((w + gap + 8, 6),
              f'{name_b}  TP={tp_b}  FP={fp_b}  (GT={n_gt})  Gains={gains}',
              fill='#FF6644', font=font_big)

    # separator line
    dr_c.line([(w, 0), (w, total_h)], fill='#666666', width=2)

    canvas.paste(im_a, (0, bar_h))
    canvas.paste(im_b, (w + gap, bar_h))

    return canvas, {'tp_a': tp_a, 'fp_a': fp_a, 'tp_b': tp_b, 'fp_b': fp_b,
                    'gains': gains, 'n_gt': n_gt}


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
    # auto screen
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f'Loading {args.name1}: {args.resume1}')
    m_a = load_model(args.config1, args.resume1, args.device)
    print(f'Loading {args.name2}: {args.resume2}')
    m_b = load_model(args.config2, args.resume2, args.device)
    print('Loading COCO ...')
    anns_by_img, imgs_info = load_coco(args.ann_file)

    # --- scan ---
    scan_n = min(args.sample_size if args.auto else args.num_images,
                 len(imgs_info))
    ids = random.sample(list(imgs_info.keys()), scan_n)

    if args.auto:
        print(f'[auto] Scanning {scan_n} images ...')
    else:
        print(f'Processing {scan_n} random images ...')

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

        tp_a = int(is_tp_a.sum())
        tp_b = int(is_tp_b.sum())

        # count true gains
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

    if args.auto:
        scored.sort(key=lambda x: (x['gains'], x['diff']), reverse=True)
        diffs = [s['diff'] for s in scored]
        gains_list = [s['gains'] for s in scored]
        print(f'\nDiff: max={max(diffs)} min={min(diffs)} mean={np.mean(diffs):.2f}')
        print(f'Gains: max={max(gains_list)} mean={np.mean(gains_list):.2f}')
        print(f'{args.name2} wins: {sum(1 for d in diffs if d > 0)}/{len(scored)}')
        top_n = min(args.top_k, len(scored))
        print(f'\nRendering top {top_n} ...')
    else:
        scored.sort(key=lambda x: (x['gains'], x['diff']), reverse=True)
        top_n = len(scored)

    for rank, s in enumerate(scored[:top_n]):
        image = Image.open(os.path.join(args.im_dir, s['info']['file_name'])).convert('RGB')
        comp, stats = draw_comparison(
            image, s['gt'],
            s['boxes_a'], s['labels_a'], s['scores_a'], s['is_tp_a'], args.name1,
            s['boxes_b'], s['labels_b'], s['scores_b'], s['is_tp_b'],
            s['gt_m_a'], args.name2)
        fname = (f'rank{rank+1:02d}_{s["img_id"]:012d}_'
                 f'dTP{s["diff"]:+d}_gains{stats["gains"]}.jpg')
        comp.save(os.path.join(args.out_dir, fname), quality=90)
        print(f'  [{rank+1:2d}] {s["info"]["file_name"]}  '
              f'TP_{args.name1}={stats["tp_a"]}/{stats["n_gt"]}  '
              f'TP_{args.name2}={stats["tp_b"]}/{stats["n_gt"]}  '
              f'ΔTP={stats["diff"]:+d}  gains={stats["gains"]}')

    print(f'\nDone → {args.out_dir}')


if __name__ == '__main__':
    main()

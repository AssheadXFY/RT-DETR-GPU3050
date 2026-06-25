"""Detection visualization & comparison for RT-DETR model variants.

Two modes:
  1. --auto : scan N images, score each by (TP_B - TP_A), keep top-K where
     model B outperforms model A most.
  2. default: process exactly --num-images random images.

Rendering modes (--render):
  tp_fp  : TP=green solid, FP=red dashed. Clearer than all-blue boxes.
  gains  : Only highlight boxes where Model B catches what Model A missed.
           GT in yellow, "gains" in bright green, both-caught in faded blue.

Usage (recommended):
    python tools/visualize_compare.py \
        -c1 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r1 ckpt/baseline.pth --name1 Baseline \
        -c2 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml -r2 ckpt/freq.pth --name2 FreqSpatial \
        --auto --sample-size 500 --top-k 15 --render gains

Requirements:
    pip install pillow pycocotools
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


# ---------------------------------------------------------------------------
# 80 COCO class names (0-indexed)
# ---------------------------------------------------------------------------

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
# model loader
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: str):
    cfg = YAMLConfig(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    cfg.model.load_state_dict(state, strict=False)

    class InferenceModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_sizes)

    wrapped = InferenceModel(cfg.model, cfg.postprocessor).to(device)
    wrapped.eval()
    return wrapped


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(model: nn.Module, image: Image.Image, device: str,
           input_size: int = 640, score_thr: float = 0.4):
    w, h = image.size
    orig_size = torch.tensor([w, h])[None].to(device)
    transform = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
    x = transform(image)[None].to(device)
    labels, boxes, scores = model(x, orig_size)

    keep = scores[0] >= score_thr
    return (boxes[0][keep].cpu().numpy(),
            labels[0][keep].cpu().numpy().astype(int),
            scores[0][keep].cpu().numpy())


# ---------------------------------------------------------------------------
# IoU & matching (returns masks, not just counts)
# ---------------------------------------------------------------------------

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-8)


def match_predictions_with_masks(pred_boxes: np.ndarray, pred_labels: np.ndarray,
                                 pred_scores: np.ndarray, gt_anns: list,
                                 iou_thr: float = 0.5):
    """Match predictions to GT. Return counts + per-prediction/per-GT masks.

    Returns:
        tp, fp, fn  -- int counts
        is_tp        -- bool[n_preds]  True if this prediction matched a GT
        gt_matched   -- bool[n_gt]     True if this GT was matched
        matched_gt_idx -- int[n_preds] GT index each prediction matched (-1=no match)
    """
    n_preds = len(pred_boxes)
    n_gt = len(gt_anns)

    is_tp = np.zeros(n_preds, dtype=bool)
    matched_gt_idx = np.full(n_preds, -1, dtype=int)

    if n_preds == 0:
        return 0, 0, n_gt, is_tp, np.zeros(n_gt, dtype=bool), matched_gt_idx

    if n_gt == 0:
        return 0, n_preds, 0, is_tp, np.zeros(0, dtype=bool), matched_gt_idx

    # GT array: [x1, y1, x2, y2, class_id]
    gt_array = np.array([
        [ann['bbox'][0], ann['bbox'][1],
         ann['bbox'][0] + ann['bbox'][2],
         ann['bbox'][1] + ann['bbox'][3],
         ann['category_id'] - 1]
        for ann in gt_anns
    ])

    # Sort predictions by score descending for COCO-style greedy matching
    order = np.argsort(-pred_scores)
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0

    for orig_idx in order:
        cls = pred_labels[orig_idx]
        candidates = [j for j in range(n_gt)
                      if not gt_matched[j] and int(gt_array[j, 4]) == cls]
        if not candidates:
            continue  # FP (is_tp stays False)

        ious = compute_iou(pred_boxes[orig_idx], gt_array[candidates])
        best_local = int(np.argmax(ious))

        if ious[best_local] >= iou_thr:
            tp += 1
            gt_matched[candidates[best_local]] = True
            is_tp[orig_idx] = True
            matched_gt_idx[orig_idx] = candidates[best_local]

    fp = n_preds - tp
    fn = int((~gt_matched).sum())
    return tp, fp, fn, is_tp, gt_matched, matched_gt_idx


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _get_font(size: int):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _draw_box(draw, x1, y1, x2, y2, color, width, label, font):
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        tb = draw.textbbox((x1, max(0, y1 - 14)), label, font=font)
        draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        draw.text((x1, max(0, y1 - 14)), label, fill='#000000', font=font)


def draw_tp_fp(image: Image.Image,
               boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray,
               is_tp: np.ndarray,
               tp_color='#00CC00', fp_color='#FF2222', width: int = 3):
    """Draw TP boxes in green, FP boxes in red dashed."""
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(12)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_id = labels[i]
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
        txt = f'{name} {scores[i]:.2f}'

        if is_tp[i]:
            _draw_box(draw, x1, y1, x2, y2, tp_color, width, txt, font)
        else:
            # red dashed — draw manually with dashed segments
            _draw_box(draw, x1, y1, x2, y2, fp_color, 2, txt, font)
    return im


def draw_gains(image: Image.Image,
               boxes_a: np.ndarray, labels_a: np.ndarray, scores_a: np.ndarray,
               boxes_b: np.ndarray, labels_b: np.ndarray, scores_b: np.ndarray,
               gt_anns: list, iou_thr: float = 0.5):
    """Highlight mode: show GT (yellow), gains (bright green), both-caught (faded blue).

    "gains" = TP in Model B but FN in Model A (i.e. Baseline missed, FreqSpatial found).
    """
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(12)

    # Match both models
    _, _, _, is_tp_a, gt_matched_a, _ = match_predictions_with_masks(
        boxes_a, labels_a, scores_a, gt_anns, iou_thr)
    _, _, _, is_tp_b, gt_matched_b, matched_gt_b = match_predictions_with_masks(
        boxes_b, labels_b, scores_b, gt_anns, iou_thr)

    # Build GT array for lookup
    gt_list = []
    for ann in gt_anns:
        x, y, w_box, h_box = ann['bbox']
        gt_list.append({
            'box': [x, y, x + w_box, y + h_box],
            'cls_id': ann['category_id'] - 1,
            'matched_a': False, 'matched_b': False,
        })
    for i in range(len(gt_list)):
        if i < len(gt_matched_a):
            gt_list[i]['matched_a'] = gt_matched_a[i]
        if i < len(gt_matched_b):
            gt_list[i]['matched_b'] = gt_matched_b[i]

    # 1. GT boxes (yellow, dashed outline)
    for gt in gt_list:
        x1, y1, x2, y2 = gt['box']
        cls_id = gt['cls_id']
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
        _draw_box(draw, x1, y1, x2, y2, '#FFD700', 2, name, font)

    # 2. Gains (FreqSpatial TP, Baseline FN) — bright green, thick
    gains = 0
    for i in range(len(boxes_b)):
        if is_tp_b[i]:
            gt_idx = matched_gt_b[i]
            if gt_idx >= 0 and not (gt_idx < len(gt_matched_a) and gt_matched_a[gt_idx]):
                # This GT was missed by Baseline => GAIN
                x1, y1, x2, y2 = boxes_b[i]
                cls_id = labels_b[i]
                name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
                _draw_box(draw, x1, y1, x2, y2, '#00FF00', 4, f'GAIN: {name}', font)
                gains += 1

    # 3. Both caught (Baseline also TP) — thin blue, no text (less clutter)
    #    Only draw if not already drawn as a gain
    for i in range(len(boxes_b)):
        if is_tp_b[i]:
            gt_idx = matched_gt_b[i]
            if gt_idx >= 0 and (gt_idx < len(gt_matched_a) and gt_matched_a[gt_idx]):
                x1, y1, x2, y2 = boxes_b[i]
                draw.rectangle([x1, y1, x2, y2], outline='#4488AA', width=1)

    return im, gains


def draw_side_by_side_tp_fp(img_a, img_b, label_a, label_b,
                            tp_a, fp_a, tp_b, fp_b, n_gt, gap=10):
    """Side-by-side with TP/FP counts overlaid."""
    w_a, h_a = img_a.size
    total_w = w_a + gap + img_b.width
    header_h = 36
    total_h = header_h + max(h_a, img_b.height)

    canvas = Image.new('RGB', (total_w, total_h), '#1a1a1a')
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)

    draw.text((8, 4), f'{label_a}  TP={tp_a}  FP={fp_a}  (GT={n_gt})',
              fill='#00CC00', font=font)
    draw.text((w_a + gap + 8, 4), f'{label_b}  TP={tp_b}  FP={fp_b}  (GT={n_gt})',
              fill='#00CC00', font=font)

    canvas.paste(img_a, (0, header_h))
    canvas.paste(img_b, (w_a + gap, header_h))
    return canvas


def draw_gt_only(image: Image.Image, gt_anns: list):
    """Draw only GT boxes in yellow on a copy of the image."""
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(14)
    for ann in gt_anns:
        x, y, w_box, h_box = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w_box, y + h_box
        cls_id = ann['category_id'] - 1
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
        draw.rectangle([x1, y1, x2, y2], outline='#FFD700', width=2)
        tb = draw.textbbox((x1, max(0, y1 - 15)), name, font=font)
        draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill='#FFD700')
        draw.text((x1, max(0, y1 - 15)), name, fill='#000000', font=font)
    return im


def draw_gains_composite(img_gt: Image.Image, img_gains: Image.Image,
                         img_b: Image.Image,
                         label_a: str, label_b: str,
                         tp_a, tp_b, n_gt):
    """Triple: GT-tagged image + gains highlight + FreqSpatial TP/FP."""
    w = img_gt.width
    gap = 8
    total_w = w * 3 + gap * 2
    header_h = 28
    total_h = header_h + max(img_gt.height, img_gains.height, img_b.height)

    canvas = Image.new('RGB', (total_w, total_h), '#1a1a1a')
    draw = ImageDraw.Draw(canvas)
    font = _get_font(13)

    draw.text((8, 4), f'Ground Truth ({n_gt} objects)', fill='#FFD700', font=font)
    draw.text((w + gap + 8, 4), f'Gains: {label_b} found, {label_a} missed',
              fill='#00FF00', font=font)
    draw.text((w * 2 + gap * 2 + 8, 4),
              f'{label_b}  TP={tp_b}/{n_gt}  (green=TP red=FP)',
              fill='#44AAFF', font=font)

    canvas.paste(img_gt, (0, header_h))
    canvas.paste(img_gains, (w + gap, header_h))
    canvas.paste(img_b, (w * 2 + gap * 2, header_h))
    return canvas


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def load_coco_annotations(ann_file: str):
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    by_image_id = defaultdict(list)
    for ann in coco['annotations']:
        by_image_id[ann['image_id']].append(ann)
    images_info = {img['id']: img for img in coco['images']}
    return by_image_id, images_info


def load_image(img_dir: str, img_info: dict) -> Image.Image:
    return Image.open(os.path.join(img_dir, img_info['file_name'])).convert('RGB')


# ---------------------------------------------------------------------------
# auto screening
# ---------------------------------------------------------------------------

def auto_screen(model_a, model_b, args, anns_by_img, images_info):
    all_img_ids = list(images_info.keys())
    sample_size = min(args.sample_size, len(all_img_ids))
    sample_ids = random.sample(all_img_ids, sample_size)

    print(f'[auto] Screening {sample_size} images ...')
    results = []

    for idx, img_id in enumerate(sample_ids):
        img_info = images_info[img_id]
        image = load_image(args.im_dir, img_info)
        gt_anns = anns_by_img.get(img_id, [])
        if not gt_anns:
            continue

        boxes_a, labels_a, scores_a = detect(
            model_a, image, args.device, score_thr=args.score_thr)
        boxes_b, labels_b, scores_b = detect(
            model_b, image, args.device, score_thr=args.score_thr)

        tp_a, fp_a, fn_a, is_tp_a, gt_m_a, _ = match_predictions_with_masks(
            boxes_a, labels_a, scores_a, gt_anns)
        tp_b, fp_b, fn_b, is_tp_b, gt_m_b, matched_b = match_predictions_with_masks(
            boxes_b, labels_b, scores_b, gt_anns)

        # Count TRUE gains: TP in B, that GT was NOT matched in A
        gain_count = 0
        for i in range(len(boxes_b)):
            if is_tp_b[i]:
                gt_idx = matched_b[i]
                if gt_idx >= 0 and not (gt_idx < len(gt_m_a) and gt_m_a[gt_idx]):
                    gain_count += 1

        results.append({
            'img_id': img_id, 'img_info': img_info,
            'boxes_a': boxes_a, 'labels_a': labels_a, 'scores_a': scores_a,
            'is_tp_a': is_tp_a,
            'boxes_b': boxes_b, 'labels_b': labels_b, 'scores_b': scores_b,
            'is_tp_b': is_tp_b,
            'matched_b': matched_b, 'gt_m_a': gt_m_a,
            'gt_anns': gt_anns,
            'tp_a': tp_a, 'fp_a': fp_a, 'fn_a': fn_a,
            'tp_b': tp_b, 'fp_b': fp_b, 'fn_b': fn_b,
            'n_gt': len(gt_anns),
            'gain_count': gain_count,
            'tp_diff': tp_b - tp_a,
        })

        if (idx + 1) % 50 == 0:
            print(f'  [{idx+1}/{sample_size}] ...')

    # Sort by gain_count first, then tp_diff
    results.sort(key=lambda r: (r['gain_count'], r['tp_diff']), reverse=True)

    all_diffs = [r['tp_diff'] for r in results]
    all_gains = [r['gain_count'] for r in results]
    print(f'\n[auto] Score distribution:')
    print(f'  TP diff: max={max(all_diffs)} min={min(all_diffs)} '
          f'mean={np.mean(all_diffs):.2f}')
    print(f'  True gains (B found, A missed): max={max(all_gains)} mean={np.mean(all_gains):.2f}')
    print(f'  Images where {args.name2} wins: {sum(1 for d in all_diffs if d > 0)}/{len(results)}')
    print(f'  Images where {args.name1} wins: {sum(1 for d in all_diffs if d < 0)}/{len(results)}')
    return results


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_results(results, args):
    top_k = min(args.top_k, len(results))
    print(f'\nRendering top {top_k} images (render mode: {args.render}) ...')

    for rank, r in enumerate(results[:top_k]):
        img_id = r['img_id']
        image = load_image(args.im_dir, r['img_info'])
        w, h = image.size

        info = (f'dTP={r["tp_diff"]:+d}  gain={r["gain_count"]}  '
                f'{args.name1} TP={r["tp_a"]}/{r["n_gt"]}  '
                f'{args.name2} TP={r["tp_b"]}/{r["n_gt"]}')

        print(f'  [rank{rank+1:02d}] {r["img_info"]["file_name"]}  {info}')

        suffix = f'rank{rank+1:02d}_{img_id:012d}_dTP{r["tp_diff"]:+d}'

        if args.render == 'gains':
            # GT image
            img_gt = draw_gt_only(image, r['gt_anns'])

            # Gains highlight
            img_gains, gain_c = draw_gains(
                image,
                r['boxes_a'], r['labels_a'], r['scores_a'],
                r['boxes_b'], r['labels_b'], r['scores_b'],
                r['gt_anns'])

            # FreqSpatial TP/FP view
            img_b = draw_tp_fp(image,
                               r['boxes_b'], r['labels_b'], r['scores_b'],
                               r['is_tp_b'],
                               tp_color='#00CC00', fp_color='#FF4444')

            # Composite: GT | Gains | FreqSpatial TP+FP
            comp = draw_gains_composite(
                img_gt, img_gains, img_b,
                args.name1, args.name2,
                r['tp_a'], r['tp_b'], r['n_gt'])
            comp.save(os.path.join(args.out_dir, f'{suffix}_gains.jpg'), quality=85)

        elif args.render == 'tp_fp':
            # TP/FP view for both models side-by-side
            img_a = draw_tp_fp(image,
                               r['boxes_a'], r['labels_a'], r['scores_a'],
                               r['is_tp_a'],
                               tp_color='#00CC00', fp_color='#FF4444')
            img_b = draw_tp_fp(image,
                               r['boxes_b'], r['labels_b'], r['scores_b'],
                               r['is_tp_b'],
                               tp_color='#00CC00', fp_color='#FF4444')
            sbs = draw_side_by_side_tp_fp(
                img_a, img_b, args.name1, args.name2,
                r['tp_a'], r['fp_a'], r['tp_b'], r['fp_b'], r['n_gt'])
            sbs.save(os.path.join(args.out_dir, f'{suffix}_tpfp.jpg'), quality=85)

        else:
            # plain mode
            img_a = draw_tp_fp(image,
                               r['boxes_a'], r['labels_a'], r['scores_a'],
                               np.ones(len(r['boxes_a']), dtype=bool),
                               tp_color='#4488FF', fp_color='#4488FF')
            img_b = draw_tp_fp(image,
                               r['boxes_b'], r['labels_b'], r['scores_b'],
                               np.ones(len(r['boxes_b']), dtype=bool),
                               tp_color='#FF6644', fp_color='#FF6644')
            from PIL import Image as PILImage
            total_w = image.width * 2 + 10
            total_h = max(image.height, img_a.height)
            canvas = PILImage.new('RGB', (total_w, total_h), '#1a1a1a')
            canvas.paste(img_a, (0, 0))
            canvas.paste(img_b, (image.width + 10, 0))
            canvas.save(os.path.join(args.out_dir, f'{suffix}_plain.jpg'), quality=85)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Detection Visual Comparison')
    parser.add_argument('-c1', '--config1', type=str, required=True)
    parser.add_argument('-r1', '--resume1', type=str, required=True)
    parser.add_argument('--name1', type=str, default='Baseline')

    parser.add_argument('-c2', '--config2', type=str, required=True)
    parser.add_argument('-r2', '--resume2', type=str, required=True)
    parser.add_argument('--name2', type=str, default='FreqSpatial')

    parser.add_argument('--im-dir', type=str, default='dataset/coco/val2017')
    parser.add_argument('--ann-file', type=str,
                        default='dataset/coco/annotations/instances_val2017.json')
    parser.add_argument('--out-dir', type=str, default='output/compare')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)

    # Rendering
    parser.add_argument('--render', type=str, default='gains',
                        choices=['gains', 'tp_fp', 'plain'],
                        help="gains: highlight what B catches that A missed | "
                             "tp_fp: green=TP red=FP for each model | "
                             "plain: all boxes one color each")

    # Manual mode
    parser.add_argument('--num-images', type=int, default=30)

    # Auto-screening
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=15)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f'Loading {args.name1}: {args.resume1}')
    model_a = load_model(args.config1, args.resume1, args.device)
    print(f'Loading {args.name2}: {args.resume2}')
    model_b = load_model(args.config2, args.resume2, args.device)
    print('Loading COCO annotations ...')
    anns_by_img, images_info = load_coco_annotations(args.ann_file)

    if args.auto:
        results = auto_screen(model_a, model_b, args, anns_by_img, images_info)
        render_results(results, args)

        print(f'\n{"="*60}')
        print(f'Top {min(args.top_k, len(results))} images:')
        for rank, r in enumerate(results[:args.top_k]):
            print(f'  {rank+1:2d}. img={r["img_id"]:012d}  '
                  f'{args.name1}_TP={r["tp_a"]}/{r["n_gt"]}  '
                  f'{args.name2}_TP={r["tp_b"]}/{r["n_gt"]}  '
                  f'ΔTP={r["tp_diff"]:+d}  gains={r["gain_count"]}')
    else:
        all_img_ids = list(images_info.keys())
        if len(all_img_ids) > args.num_images:
            sample_ids = random.sample(all_img_ids, args.num_images)
        else:
            sample_ids = all_img_ids

        print(f'Processing {len(sample_ids)} random images ...')
        results_list = []
        for idx, img_id in enumerate(sample_ids):
            img_info = images_info[img_id]
            image = load_image(args.im_dir, img_info)
            gt_anns = anns_by_img.get(img_id, [])
            print(f'  [{idx+1}/{len(sample_ids)}] {img_info["file_name"]}')

            boxes_a, labels_a, scores_a = detect(
                model_a, image, args.device, score_thr=args.score_thr)
            boxes_b, labels_b, scores_b = detect(
                model_b, image, args.device, score_thr=args.score_thr)
            tp_a, fp_a, _, is_tp_a, gt_m_a, _ = match_predictions_with_masks(
                boxes_a, labels_a, scores_a, gt_anns)
            tp_b, fp_b, _, is_tp_b, gt_m_b, matched_b = match_predictions_with_masks(
                boxes_b, labels_b, scores_b, gt_anns)

            gain_count = 0
            for i in range(len(boxes_b)):
                if is_tp_b[i]:
                    gt_idx = matched_b[i]
                    if gt_idx >= 0 and not (gt_idx < len(gt_m_a) and gt_m_a[gt_idx]):
                        gain_count += 1

            results_list.append({
                'img_id': img_id, 'img_info': img_info,
                'boxes_a': boxes_a, 'labels_a': labels_a, 'scores_a': scores_a,
                'is_tp_a': is_tp_a,
                'boxes_b': boxes_b, 'labels_b': labels_b, 'scores_b': scores_b,
                'is_tp_b': is_tp_b,
                'matched_b': matched_b, 'gt_m_a': gt_m_a,
                'gt_anns': gt_anns,
                'tp_a': tp_a, 'fp_a': fp_a,
                'tp_b': tp_b, 'fp_b': fp_b,
                'n_gt': len(gt_anns),
                'gain_count': gain_count,
                'tp_diff': tp_b - tp_a,
            })

        results_list.sort(key=lambda r: (r['gain_count'], r['tp_diff']), reverse=True)
        render_results(results_list, args)

    print(f'\nDone. Output directory: {args.out_dir}')


if __name__ == '__main__':
    main()

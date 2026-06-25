"""Detection visualization & comparison for RT-DETR model variants.

Two modes:
  1. --auto : scan N images, score each by (TP_B - TP_A), keep top-K where
     model B outperforms model A most. Best for finding paper-worthy cases.
  2. default: process exactly --num-images random images.

Usage (auto-screening recommended):
    python tools/visualize_compare.py \
        -c1 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r1 ckpt/baseline.pth --name1 Baseline \
        -c2 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml -r2 ckpt/freq.pth --name2 FreqSpatial \
        --auto --sample-size 500 --top-k 20

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
    """Load a RT-DETR model + postprocessor in deploy mode."""
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
    """Run detection on a single PIL image.

    Returns:
        boxes_xyxy  numpy (N,4)  -- absolute pixel coords
        labels      numpy (N,)   -- 0-indexed class ids
        scores      numpy (N,)   -- confidence in [0,1]
    """
    w, h = image.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transform = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
    x = transform(image)[None].to(device)

    labels, boxes, scores = model(x, orig_size)

    keep = scores[0] >= score_thr
    boxes_xyxy = boxes[0][keep].cpu().numpy()
    _labels    = labels[0][keep].cpu().numpy().astype(int)
    _scores    = scores[0][keep].cpu().numpy()

    return boxes_xyxy, _labels, _scores


# ---------------------------------------------------------------------------
# scoring: per-image TP/FP/FN
# ---------------------------------------------------------------------------

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against many boxes (M,4), all xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-8)


def match_predictions(pred_boxes: np.ndarray, pred_labels: np.ndarray,
                      pred_scores: np.ndarray,
                      gt_anns: list, iou_thr: float = 0.5):
    """Match predictions to ground-truth boxes.

    Returns:
        tp  int   -- number of GT boxes correctly detected
        fp  int   -- number of predictions with no matching GT
        fn  int   -- number of GT boxes not detected
    """
    if len(pred_boxes) == 0 and len(gt_anns) == 0:
        return 0, 0, 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_anns)
    if len(gt_anns) == 0:
        return 0, len(pred_boxes), 0

    # Build GT array: (M, 5) [x1,y1,x2,y2,class_id]
    gt_array = []
    for ann in gt_anns:
        x, y, w_box, h_box = ann['bbox']
        gt_array.append([x, y, x + w_box, y + h_box, ann['category_id'] - 1])
    gt_array = np.array(gt_array)

    # Sort predictions by score descending (COCO-style greedy matching)
    order = np.argsort(-pred_scores)
    pred_boxes_s = pred_boxes[order]
    pred_labels_s = pred_labels[order]

    gt_matched = np.zeros(len(gt_array), dtype=bool)
    tp = 0
    fp = 0

    for i in range(len(pred_boxes_s)):
        cls = pred_labels_s[i]
        # Candidates: same class + not yet matched
        candidates = [j for j in range(len(gt_array))
                      if not gt_matched[j] and int(gt_array[j, 4]) == cls]
        if not candidates:
            fp += 1
            continue

        ious = compute_iou(pred_boxes_s[i], gt_array[candidates])
        best_j = candidates[int(np.argmax(ious))]

        if ious.max() >= iou_thr:
            tp += 1
            gt_matched[best_j] = True
        else:
            fp += 1

    fn = int((~gt_matched).sum())
    return tp, fp, fn


def score_image(pred_boxes_a, pred_labels_a, pred_scores_a,
                pred_boxes_b, pred_labels_b, pred_scores_b,
                gt_anns: list, iou_thr: float = 0.5):
    """Return a dict with per-image metrics for two models."""
    tp_a, fp_a, fn_a = match_predictions(
        pred_boxes_a, pred_labels_a, pred_scores_a, gt_anns, iou_thr)
    tp_b, fp_b, fn_b = match_predictions(
        pred_boxes_b, pred_labels_b, pred_scores_b, gt_anns, iou_thr)

    n_gt = len(gt_anns)
    recall_a = tp_a / max(n_gt, 1)
    recall_b = tp_b / max(n_gt, 1)
    precision_a = tp_a / max(tp_a + fp_a, 1)
    precision_b = tp_b / max(tp_b + fp_b, 1)

    return {
        'tp_a': tp_a, 'fp_a': fp_a, 'fn_a': fn_a,
        'tp_b': tp_b, 'fp_b': fp_b, 'fn_b': fn_b,
        'n_gt': n_gt,
        'recall_a': recall_a, 'recall_b': recall_b,
        'precision_a': precision_a, 'precision_b': precision_b,
        'tp_diff': tp_b - tp_a,        # positive = B wins
        'recall_diff': recall_b - recall_a,
    }


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def make_label_text(cls_id: int, score: float) -> str:
    name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
    return f'{name} {score:.2f}'


def _get_font(size: int):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes, labels, scores,
               color='#00FF00', width: int = 3, font_size: int = 16):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(font_size)

    for box, cls_id, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = make_label_text(cls_id, score)
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        draw.text((x1, y1), text, fill='#FFFFFF', font=font)
    return im


def draw_groundtruth(image: Image.Image, annotations: list,
                     color='#FFFF00', width: int = 2):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font(14)

    for ann in annotations:
        x, y, w_box, h_box = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w_box, y + h_box
        cls_id = ann['category_id'] - 1
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 14)), name, fill=color, font=font)
    return im


def make_side_by_side(img_a: Image.Image, img_b: Image.Image,
                      label_a: str, label_b: str,
                      gap: int = 10, header_height: int = 32,
                      bg: str = '#1a1a1a', score_diff: str = None) -> Image.Image:
    w_a, h_a = img_a.size
    w_b, h_b = img_b.size
    total_w = w_a + gap + w_b
    total_h = header_height + max(h_a, h_b)

    canvas = Image.new('RGB', (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)
    font = _get_font(16)
    font_small = _get_font(12)

    draw.text((10, 4), label_a, fill='#FFFFFF', font=font)
    draw.text((w_a + gap + 10, 4), label_b, fill='#FFFFFF', font=font)

    if score_diff:
        draw.text((total_w // 2 - 40, 16), score_diff, fill='#FFD700', font=font_small)

    canvas.paste(img_a, (0, header_height))
    canvas.paste(img_b, (w_a + gap, header_height))
    return canvas


def make_triple(gt_im: Image.Image, img_a: Image.Image, img_b: Image.Image,
                label_a: str, label_b: str, score_diff: str = None) -> Image.Image:
    row2 = make_side_by_side(img_a, img_b, label_a, label_b,
                             header_height=24, gap=8, bg='#2a2a2a',
                             score_diff=score_diff)

    total_w = max(gt_im.width, row2.width)
    total_h = gt_im.height + 8 + row2.height

    canvas = Image.new('RGB', (total_w, total_h), '#1a1a1a')
    # GT label
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)
    draw.text(((total_w - gt_im.width) // 2 + 4, 2), 'Ground Truth',
              fill='#FFFF00', font=font)

    canvas.paste(gt_im, ((total_w - gt_im.width) // 2, 18))
    canvas.paste(row2, ((total_w - row2.width) // 2, gt_im.height + 8))
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
    """Scan sample_size images, score each, sort by tp_diff, return top-K results."""
    all_img_ids = list(images_info.keys())
    sample_size = min(args.sample_size, len(all_img_ids))
    sample_ids = random.sample(all_img_ids, sample_size)

    print(f'[auto] Screening {sample_size} images to find best comparisons ...')
    results = []

    for idx, img_id in enumerate(sample_ids):
        img_info = images_info[img_id]
        image = load_image(args.im_dir, img_info)
        gt_anns = anns_by_img.get(img_id, [])

        if not gt_anns:
            continue  # skip images without objects

        boxes_a, labels_a, scores_a = detect(
            model_a, image, args.device, score_thr=args.score_thr)
        boxes_b, labels_b, scores_b = detect(
            model_b, image, args.device, score_thr=args.score_thr)

        sc = score_image(boxes_a, labels_a, scores_a,
                         boxes_b, labels_b, scores_b, gt_anns)

        results.append({
            'img_id': img_id,
            'img_info': img_info,
            'boxes_a': boxes_a, 'labels_a': labels_a, 'scores_a': scores_a,
            'boxes_b': boxes_b, 'labels_b': labels_b, 'scores_b': scores_b,
            'gt_anns': gt_anns,
            'score': sc,
        })

        if (idx + 1) % 50 == 0:
            print(f'  [{idx+1}/{sample_size}] scanned ...')

    # Sort by tp_diff descending (B wins most at top)
    results.sort(key=lambda r: r['score']['tp_diff'], reverse=True)

    # Print summary
    print(f'\n[auto] Score distribution (tp_diff = TP_{args.name2} - TP_{args.name1}):')
    all_diffs = [r['score']['tp_diff'] for r in results]
    print(f'  Max: {max(all_diffs)}, Min: {min(all_diffs)}, '
          f'Mean: {np.mean(all_diffs):.2f}, Std: {np.std(all_diffs):.2f}')
    print(f'  Images where {args.name2} wins (tp_diff>0): '
          f'{sum(1 for d in all_diffs if d > 0)}/{len(results)}')
    print(f'  Images where {args.name1} wins (tp_diff<0): '
          f'{sum(1 for d in all_diffs if d < 0)}/{len(results)}')
    print(f'  Images tied (tp_diff=0): {sum(1 for d in all_diffs if d == 0)}/{len(results)}')

    return results


def render_results(results, args):
    """Render selected results to disk."""
    top_k = min(args.top_k, len(results))
    print(f'\nRendering top {top_k} images ...')

    for rank, r in enumerate(results[:top_k]):
        img_id = r['img_id']
        sc = r['score']

        image = load_image(args.im_dir, r['img_info'])

        # Generate images
        img_a = draw_boxes(image, r['boxes_a'], r['labels_a'], r['scores_a'],
                           color='#00BFFF')
        img_b = draw_boxes(image, r['boxes_b'], r['labels_b'], r['scores_b'],
                           color='#FF4444')
        img_gt = draw_groundtruth(image, r['gt_anns'], color='#FFFF00')

        diff_str = (f"ΔTP = +{sc['tp_diff']}  "
                    f"({args.name1}: {sc['tp_a']}/{sc['n_gt']}  "
                    f"{args.name2}: {sc['tp_b']}/{sc['n_gt']})")

        # side-by-side
        sbs = make_side_by_side(img_a, img_b, args.name1, args.name2,
                                score_diff=diff_str)
        out_sbs = f'rank{rank+1:02d}_{img_id:012d}_tp{sc["tp_diff"]:+d}_sbs.jpg'
        sbs.save(os.path.join(args.out_dir, out_sbs), quality=85)

        # triple
        triple = make_triple(img_gt, img_a, img_b, args.name1, args.name2,
                             score_diff=diff_str)
        out_tri = f'rank{rank+1:02d}_{img_id:012d}_tp{sc["tp_diff"]:+d}_triple.jpg'
        triple.save(os.path.join(args.out_dir, out_tri), quality=85)

        print(f'  [rank{rank+1:02d}] {r["img_info"]["file_name"]}  '
              f'{args.name1}_TP={sc["tp_a"]}  {args.name2}_TP={sc["tp_b"]}  '
              f'ΔTP={sc["tp_diff"]:+d}  GT={sc["n_gt"]}')


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

    # Manual mode
    parser.add_argument('--num-images', type=int, default=30,
                        help='(manual mode) Number of random images')
    parser.add_argument('--seed', type=int, default=42)

    # Auto-screening mode
    parser.add_argument('--auto', action='store_true',
                        help='Enable auto-screening: scan many images, '
                             'pick top-K where model B outperforms model A most')
    parser.add_argument('--sample-size', type=int, default=500,
                        help='[auto mode] How many validation images to scan')
    parser.add_argument('--top-k', type=int, default=20,
                        help='[auto mode] How many best images to render')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Load models
    print(f'Loading model A ({args.name1}): {args.resume1}')
    model_a = load_model(args.config1, args.resume1, args.device)
    print(f'Loading model B ({args.name2}): {args.resume2}')
    model_b = load_model(args.config2, args.resume2, args.device)

    print('Loading COCO annotations ...')
    anns_by_img, images_info = load_coco_annotations(args.ann_file)

    if args.auto:
        # -- Auto-screening mode --
        results = auto_screen(model_a, model_b, args, anns_by_img, images_info)
        render_results(results, args)

        # Stats summary
        print(f'\n{"="*60}')
        print(f'Score summary (top {min(args.top_k, len(results))}):')
        for rank, r in enumerate(results[:args.top_k]):
            sc = r['score']
            print(f'  {rank+1:2d}. img={r["img_id"]:012d}  '
                  f'{args.name1}_TP={sc["tp_a"]}/{sc["n_gt"]}  '
                  f'{args.name2}_TP={sc["tp_b"]}/{sc["n_gt"]}  '
                  f'ΔTP={sc["tp_diff"]:+d}')
        print(f'{args.name2} > {args.name1} in '
              f'{sum(1 for r in results if r["score"]["tp_diff"] > 0)}/{len(results)} images')
    else:
        # -- Manual random mode --
        all_img_ids = list(images_info.keys())
        if len(all_img_ids) > args.num_images:
            sample_ids = random.sample(all_img_ids, args.num_images)
        else:
            sample_ids = all_img_ids

        print(f'Processing {len(sample_ids)} random images ...')
        for idx, img_id in enumerate(sample_ids):
            img_info = images_info[img_id]
            image = load_image(args.im_dir, img_info)
            gt_anns = anns_by_img.get(img_id, [])

            print(f'  [{idx+1}/{len(sample_ids)}] {img_info["file_name"]}')

            boxes_a, labels_a, scores_a = detect(
                model_a, image, args.device, score_thr=args.score_thr)
            boxes_b, labels_b, scores_b = detect(
                model_b, image, args.device, score_thr=args.score_thr)

            img_a = draw_boxes(image, boxes_a, labels_a, scores_a, color='#00BFFF')
            img_b = draw_boxes(image, boxes_b, labels_b, scores_b, color='#FF4444')
            img_gt = draw_groundtruth(image, gt_anns, color='#FFFF00')

            sbs = make_side_by_side(img_a, img_b, args.name1, args.name2)
            sbs.save(os.path.join(args.out_dir, f'{img_id:012d}_sbs.jpg'), quality=85)

            triple = make_triple(img_gt, img_a, img_b, args.name1, args.name2)
            triple.save(os.path.join(args.out_dir, f'{img_id:012d}_triple.jpg'),
                        quality=85)

    print(f'\nDone. Output directory: {args.out_dir}')


if __name__ == '__main__':
    main()

"""Detection visualization & comparison for RT-DETR model variants.

Loads checkpoints for two (or more) model variants, runs inference on COCO
validation images, and produces side-by-side comparison images with bounding
boxes, class labels, and confidence scores.

Usage:
    python tools/visualize_compare.py \
        -c1 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r1 /path/to/baseline.pth --name1 Baseline \
        -c2 configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_freq.yml -r2 /path/to/freq.pth --name2 FreqSpatial \
        --im-dir dataset/coco/val2017 --ann-file dataset/coco/annotations/instances_val2017.json \
        --out-dir output/compare --score-thr 0.4 --num-images 50

Requirements:
    pip install pillow pycocotools

AUTHOR:   (c) generated
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

# Distinct colors (cycled)
COLORS = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231',
    '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
    '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC',
    '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7',
]


# ---------------------------------------------------------------------------
# model loader
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load a RT-DETR model + postprocessor in deploy mode."""
    cfg = YAMLConfig(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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
    x = transform(image)[None].to(device)  # [1, 3, H, W]

    labels, boxes, scores = model(x, orig_size)

    # filter
    keep = scores[0] >= score_thr
    boxes_xyxy = boxes[0][keep].cpu().numpy()       # (K, 4)
    _labels    = labels[0][keep].cpu().numpy().astype(int)  # (K,)
    _scores    = scores[0][keep].cpu().numpy()           # (K,)

    return boxes_xyxy, _labels, _scores


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def make_label_text(cls_id: int, score: float) -> str:
    name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
    return f'{name} {score:.2f}'


def draw_boxes(image: Image.Image, boxes, labels, scores,
               color='#00FF00', width: int = 3, font_size: int = 16):
    """Draw bounding boxes on a copy of the image."""
    im = image.copy()
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()

    for box, cls_id, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        # box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        # label background
        text = make_label_text(cls_id, score)
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)
        # label text
        draw.text((x1, y1), text, fill='#FFFFFF', font=font)

    return im


def draw_groundtruth(image: Image.Image, annotations: list,
                     color='#FFFF00', width: int = 2):
    """Draw COCO ground-truth boxes in yellow."""
    im = image.copy()
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for ann in annotations:
        x, y, w_box, h_box = ann['bbox']  # COCO bbox = [x, y, w, h]
        x1, y1, x2, y2 = x, y, x + w_box, y + h_box
        cls_id = ann['category_id'] - 1  # COCO 1..90 -> 0..79
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls{cls_id}'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 14)), name, fill=color, font=font)

    return im


def make_side_by_side(img_a: Image.Image, img_b: Image.Image,
                      label_a: str, label_b: str,
                      gap: int = 10, header_height: int = 30,
                      bg: str = '#1a1a1a') -> Image.Image:
    """Combine two images side-by-side with title headers."""
    w_a, h_a = img_a.size
    w_b, h_b = img_b.size
    total_w = w_a + gap + w_b
    total_h = header_height + max(h_a, h_b)

    canvas = Image.new('RGB', (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font = ImageFont.load_default()

    # headers
    draw.text((10, 5), label_a, fill='#FFFFFF', font=font)
    draw.text((w_a + gap + 10, 5), label_b, fill='#FFFFFF', font=font)

    canvas.paste(img_a, (0, header_height))
    canvas.paste(img_b, (w_a + gap, header_height))

    return canvas


def make_triple(gt_im: Image.Image, img_a: Image.Image, img_b: Image.Image,
                label_a: str, label_b: str) -> Image.Image:
    """Ground truth on left, two predictions side-by-side on right (stacks)."""
    w_gt, h_gt = gt_im.size
    w_a, h_a = img_a.size

    row2 = make_side_by_side(img_a, img_b, label_a, label_b,
                             header_height=24, gap=8, bg='#2a2a2a')

    total_w = max(w_gt, row2.width)
    total_h = h_gt + 8 + row2.height

    canvas = Image.new('RGB', (total_w, total_h), '#1a1a1a')
    canvas.paste(gt_im, ((total_w - w_gt) // 2, 0))
    canvas.paste(row2, ((total_w - row2.width) // 2, h_gt + 8))
    return canvas


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def load_coco_annotations(ann_file: str):
    """Load COCO annotation dict, indexed by image_id."""
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
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Detection Visual Comparison')
    parser.add_argument('-c1', '--config1', type=str, required=True,
                        help='Config YAML for model A (e.g. Baseline)')
    parser.add_argument('-r1', '--resume1', type=str, required=True,
                        help='Checkpoint .pth for model A')
    parser.add_argument('--name1', type=str, default='Baseline')

    parser.add_argument('-c2', '--config2', type=str, required=True,
                        help='Config YAML for model B (e.g. FreqSpatial)')
    parser.add_argument('-r2', '--resume2', type=str, required=True,
                        help='Checkpoint .pth for model B')
    parser.add_argument('--name2', type=str, default='FreqSpatial')

    parser.add_argument('--im-dir', type=str, default='dataset/coco/val2017',
                        help='COCO val2017 image directory')
    parser.add_argument('--ann-file', type=str,
                        default='dataset/coco/annotations/instances_val2017.json',
                        help='COCO annotation JSON')
    parser.add_argument('--out-dir', type=str, default='output/compare',
                        help='Output directory for comparison images')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.4,
                        help='Confidence score threshold')
    parser.add_argument('--num-images', type=int, default=30,
                        help='Number of images to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for image selection')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Load models
    print(f'Loading model A ({args.name1}): {args.resume1}')
    model_a = load_model(args.config1, args.resume1, args.device)

    print(f'Loading model B ({args.name2}): {args.resume2}')
    model_b = load_model(args.config2, args.resume2, args.device)

    # Load COCO annotations
    print('Loading COCO annotations ...')
    anns_by_img, images_info = load_coco_annotations(args.ann_file)

    # Sample images
    all_img_ids = list(images_info.keys())
    if len(all_img_ids) > args.num_images:
        sample_ids = random.sample(all_img_ids, args.num_images)
    else:
        sample_ids = all_img_ids

    print(f'Processing {len(sample_ids)} images ...')

    for idx, img_id in enumerate(sample_ids):
        img_info = images_info[img_id]
        img_file = img_info['file_name']
        image = load_image(args.im_dir, img_info)
        gt_anns = anns_by_img.get(img_id, [])

        print(f'  [{idx+1}/{len(sample_ids)}] {img_file}')

        # --- Model A ---
        boxes_a, labels_a, scores_a = detect(
            model_a, image, args.device, score_thr=args.score_thr)
        img_a = draw_boxes(image, boxes_a, labels_a, scores_a, color='#00BFFF')

        # --- Model B ---
        boxes_b, labels_b, scores_b = detect(
            model_b, image, args.device, score_thr=args.score_thr)
        img_b = draw_boxes(image, boxes_b, labels_b, scores_b, color='#FF4444')

        # --- GT ---
        img_gt = draw_groundtruth(image, gt_anns, color='#FFFF00')

        # --- Side-by-side comparison (A | B) ---
        sbs = make_side_by_side(img_a, img_b, args.name1, args.name2)
        out_name = f'{img_id:012d}_sbs.jpg'
        sbs.save(os.path.join(args.out_dir, out_name), quality=85)

        # --- Triple layout (GT | A | B) ---
        header_w = max(img_gt.width, img_a.width, img_b.width)
        total_w_sbs = img_a.width + 10 + img_b.width
        # Two rows: GT on top row, A|B side-by-side on bottom row
        triple = make_triple(img_gt, img_a, img_b, args.name1, args.name2)
        out_name_triple = f'{img_id:012d}_triple.jpg'
        triple.save(os.path.join(args.out_dir, out_name_triple), quality=85)

    # Summary
    print(f'\nDone. Comparison images saved to: {args.out_dir}')
    print(f'  *_sbs.jpg    -- {args.name1} vs {args.name2} (side-by-side)')
    print(f'  *_triple.jpg -- Ground Truth + {args.name1} + {args.name2}')


if __name__ == '__main__':
    main()

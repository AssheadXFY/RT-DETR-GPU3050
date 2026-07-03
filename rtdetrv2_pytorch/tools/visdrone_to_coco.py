"""Convert VisDrone2019-DET annotations to COCO format.

Usage:
  python tools/visdrone_to_coco.py --visdrone_root /path/to/visdrone --out_dir /path/to/output
"""
import os, json, argparse
from PIL import Image

VISDRONE_CLASSES = [
    'ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
]
# Skip class 0 (ignored) for detection, remap to 1-based → 0-based
CLASS_REMAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}
COCO_CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"},
]


def convert_split(visdrone_root, split_name, out_dir):
    img_dir = os.path.join(visdrone_root, f"VisDrone2019-DET-{split_name}", "images")
    ann_dir = os.path.join(visdrone_root, f"VisDrone2019-DET-{split_name}", "annotations")

    os.makedirs(out_dir, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, img_name.rsplit('.', 1)[0] + '.txt')

        try:
            w, h = Image.open(img_path).size
        except Exception:
            w, h = 2000, 1500

        img_id = len(images)
        images.append({
            "id": img_id,
            "file_name": os.path.join(
                f"VisDrone2019-DET-{split_name}", "images", img_name),
            "width": w, "height": h,
        })

        if not os.path.exists(ann_path):
            continue

        for line in open(ann_path):
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            cls = int(parts[5])
            if cls not in CLASS_REMAP:
                continue
            x1, y1, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            if bw <= 0 or bh <= 0:
                continue
            # convert to coco bbox [x, y, w, h] (absolute pixels, xywh)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_REMAP[cls],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": COCO_CATEGORIES,
    }
    out_path = os.path.join(out_dir, f"visdrone_{split_name}.json")
    json.dump(out, open(out_path, 'w'))
    print(f"{split_name}: {len(images)} images, {len(annotations)} annotations → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visdrone_root', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    for s in ['train', 'val']:
        convert_split(args.visdrone_root, s, args.out_dir)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
"""Convert SWRD YOLO box annotations to MMSegmentation mask annotations.

The source dataset is expected to have the following structure:

    yolo/
      train.txt
      val.txt
      images/train2021/*.jpg
      images/val2021/*.jpg
      labels/train2021/*.txt
      labels/val2021/*.txt

The converter writes semantic masks to:

    yolo/annotations/train2021/*.png
    yolo/annotations/val2021/*.png

Mask label 0 is background. YOLO class ids 0-7 are shifted to mask labels 1-8.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = {
    'train': 'train2021',
    'val': 'val2021',
}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'root',
        type=Path,
        help='Path to the SWRD YOLO dataset root.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing generated masks.')
    return parser.parse_args()


def find_image(image_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        image_path = image_dir / f'{stem}{suffix}'
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f'Cannot find image for {stem} in {image_dir}')


def read_split(root: Path, split: str, split_dir: str) -> list[str]:
    split_file = root / f'{split}.txt'
    if split_file.exists():
        return [
            line.strip()
            for line in split_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    image_dir = root / 'images' / split_dir
    return sorted(p.stem for p in image_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def yolo_line_to_xyxy(line: str, width: int, height: int,
                      label_path: Path) -> tuple[int, int, int, int, int]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f'{label_path} should contain 5 columns per line')

    class_id = int(float(parts[0]))
    cx, cy, box_w, box_h = [float(x) for x in parts[1:]]
    x1 = int(round((cx - box_w / 2) * width))
    y1 = int(round((cy - box_h / 2) * height))
    x2 = int(round((cx + box_w / 2) * width))
    y2 = int(round((cy + box_h / 2) * height))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return class_id + 1, x1, y1, x2, y2


def convert_one(image_path: Path, label_path: Path, mask_path: Path,
                overwrite: bool) -> int:
    if mask_path.exists() and not overwrite:
        return 0

    with Image.open(image_path) as image:
        width, height = image.size

    mask = np.zeros((height, width), dtype=np.uint8)
    if label_path.exists():
        lines = [
            line.strip()
            for line in label_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        for line in lines:
            label, x1, y1, x2, y2 = yolo_line_to_xyxy(
                line, width, height, label_path)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = label

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode='L').save(mask_path)
    return 1


def convert(root: Path, overwrite: bool = False) -> dict:
    root = root.expanduser().resolve()
    summary = {}
    for split, split_dir in SPLITS.items():
        stems = read_split(root, split, split_dir)
        image_dir = root / 'images' / split_dir
        label_dir = root / 'labels' / split_dir
        mask_dir = root / 'annotations' / split_dir

        converted = 0
        for stem in stems:
            image_path = find_image(image_dir, stem)
            label_path = label_dir / f'{stem}.txt'
            mask_path = mask_dir / f'{stem}.png'
            converted += convert_one(image_path, label_path, mask_path,
                                     overwrite)
        summary[split] = dict(samples=len(stems), converted=converted)
    return summary


def main():
    args = parse_args()
    summary = convert(args.root, args.overwrite)
    for split, item in summary.items():
        print(f'{split}: {item["samples"]} samples, '
              f'{item["converted"]} masks written')


if __name__ == '__main__':
    main()

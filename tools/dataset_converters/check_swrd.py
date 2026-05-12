# Copyright (c) OpenMMLab. All rights reserved.
"""Check SWRD masks generated from YOLO boxes.

Run after ``tools/dataset_converters/swrd.py`` to verify that the coarse
semantic segmentation masks match the image split sizes and contain only valid
labels.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from swrd import convert


SPLITS = {
    'train': 'train2021',
    'val': 'val2021',
}
EXPECTED_COUNTS = {
    'train': 2726,
    'val': 682,
}
VALID_LABELS = set(range(9))
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Path to SWRD YOLO root.')
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Generate missing masks before checking.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite masks when used with --convert.')
    parser.add_argument(
        '--skip-expected-counts',
        action='store_true',
        help='Do not require the known 2726/682 split sizes.')
    return parser.parse_args()


def count_images(image_dir: Path) -> int:
    return sum(1 for p in image_dir.iterdir()
               if p.suffix.lower() in IMAGE_EXTENSIONS)


def check_split(root: Path, split: str, split_dir: str,
                skip_expected_counts: bool) -> dict:
    image_dir = root / 'images' / split_dir
    mask_dir = root / 'annotations' / split_dir
    image_count = count_images(image_dir)
    mask_paths = sorted(mask_dir.glob('*.png'))
    mask_count = len(mask_paths)

    if not skip_expected_counts:
        expected = EXPECTED_COUNTS[split]
        if image_count != expected:
            raise AssertionError(
                f'{split} image count {image_count} != expected {expected}')

    if image_count != mask_count:
        raise AssertionError(
            f'{split} image/mask count mismatch: {image_count}/{mask_count}')

    labels = set()
    for mask_path in mask_paths:
        labels.update(np.unique(np.asarray(Image.open(mask_path))).tolist())

    invalid = labels - VALID_LABELS
    if invalid:
        raise AssertionError(f'{split} has invalid mask labels: {invalid}')

    return dict(images=image_count, masks=mask_count, labels=sorted(labels))


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.convert:
        summary = convert(root, overwrite=args.overwrite)
        for split, item in summary.items():
            print(f'{split}: {item["samples"]} samples, '
                  f'{item["converted"]} masks written')

    for split, split_dir in SPLITS.items():
        item = check_split(root, split, split_dir, args.skip_expected_counts)
        print(f'{split}: {item["images"]} images, {item["masks"]} masks, '
              f'labels={item["labels"]}')

    print('SWRD check passed.')


if __name__ == '__main__':
    main()

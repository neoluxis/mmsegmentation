# Copyright (c) OpenMMLab. All rights reserved.
"""Run cross-dataset inference on the RIAWELC classification-style folders.

The current RIAWELC copy has image folders such as ``Difetto1`` and
``NoDifetto`` but no pixel-level masks. This script saves qualitative overlays
and CSV files with predicted pixel counts per semantic class.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path

import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist

if not hasattr(np, 'sctypes'):
    np.sctypes = dict(
        float=[np.float16, np.float32, np.float64],
        int=[np.int8, np.int16, np.int32, np.int64],
        uint=[np.uint8, np.uint16, np.uint32, np.uint64],
        complex=[np.complex64, np.complex128],
        others=[np.bool_, np.object_, np.bytes_, np.str_])
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = np.bool_

from mmseg.apis import inference_model, init_model, show_result_pyplot


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('config', help='MMSeg config path.')
    parser.add_argument('checkpoint', help='Checkpoint path.')
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('../datasets/RIAWELC/DB/testing'),
        help='RIAWELC split root with class subfolders.')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('work_dirs/riawelc_infer'),
        help='Directory for overlays and summary.csv.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opacity', type=float, default=0.5)
    parser.add_argument(
        '--no-overlays',
        action='store_true',
        help='Skip qualitative overlay images and only write CSV summaries.')
    parser.add_argument(
        '--write-masks',
        action='store_true',
        help='Save the uint8 predicted semantic mask for every image.')
    parser.add_argument(
        '--overlay-sample-per-folder',
        type=int,
        default=0,
        help=('Save sampled overlays per RIAWELC folder. A value of 20 keeps '
              '8 high-response, 6 mid-response, and 6 low-response images. '
              'Use 0 to save overlays for every image when overlays are on.'))
    return parser.parse_args()


def iter_images(data_root: Path):
    for image_path in sorted(data_root.rglob('*')):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def select_overlay_rows(rows: list[dict], per_folder: int) -> list[dict]:
    selected = []
    folders = sorted({row['folder'] for row in rows})
    for folder in folders:
        folder_rows = sorted(
            [row for row in rows if row['folder'] == folder],
            key=lambda item: item['defect_ratio'])
        if per_folder <= 0 or len(folder_rows) <= per_folder:
            selected.extend(folder_rows)
            continue

        high_count = min(8, per_folder)
        remaining = per_folder - high_count
        mid_count = min(6, remaining)
        low_count = max(0, remaining - mid_count)

        picks = []
        picks.extend(folder_rows[:low_count])
        if mid_count:
            mid_start = max(0, (len(folder_rows) - mid_count) // 2)
            picks.extend(folder_rows[mid_start:mid_start + mid_count])
        picks.extend(folder_rows[-high_count:])

        deduped = []
        seen = set()
        for row in picks:
            if row['relative_path'] in seen:
                continue
            seen.add(row['relative_path'])
            deduped.append(row)

        if len(deduped) < per_folder:
            for row in folder_rows:
                if row['relative_path'] in seen:
                    continue
                seen.add(row['relative_path'])
                deduped.append(row)
                if len(deduped) == per_folder:
                    break
        selected.extend(deduped[:per_folder])
    return selected


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    overlay_dir = out_dir / 'overlays'
    mask_dir = out_dir / 'masks'
    mkdir_or_exist(out_dir)
    if not args.no_overlays:
        mkdir_or_exist(overlay_dir)
    if args.write_masks:
        mkdir_or_exist(mask_dir)

    model = init_model(args.config, args.checkpoint, device=args.device)
    classes = model.dataset_meta['classes']
    header = [
        'relative_path', 'folder', 'total_pixels', 'defect_pixels',
        'defect_ratio'
    ] + [
        f'pixels_{name}' for name in classes
    ]
    rows = []
    row_records = []
    folder_totals = {}
    folder_images = {}

    for image_path in iter_images(data_root):
        result = inference_model(model, str(image_path))
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
        counts = np.bincount(pred.reshape(-1), minlength=len(classes))
        relative_path = image_path.relative_to(data_root)
        folder = relative_path.parts[0]
        total_pixels = int(counts[:len(classes)].sum())
        defect_pixels = int(counts[1:len(classes)].sum())
        defect_ratio = defect_pixels / total_pixels if total_pixels else 0.0

        if args.write_masks:
            mask_path = mask_dir / relative_path.with_suffix('.png')
            mkdir_or_exist(mask_path.parent)
            mmcv.imwrite(pred.astype(np.uint8), str(mask_path))

        if not args.no_overlays and args.overlay_sample_per_folder <= 0:
            out_file = overlay_dir / relative_path.with_suffix('.png')
            mkdir_or_exist(out_file.parent)

            vis = show_result_pyplot(
                model,
                str(image_path),
                result,
                opacity=args.opacity,
                title=str(relative_path),
                draw_gt=False,
                draw_pred=True,
                show=False,
                with_labels=False)
            mmcv.imwrite(mmcv.rgb2bgr(vis), str(out_file))

        folder_totals.setdefault(
            folder, np.zeros(len(classes), dtype=np.int64))
        folder_totals[folder] += counts[:len(classes)]
        folder_images[folder] = folder_images.get(folder, 0) + 1

        row = [
            str(relative_path), folder, str(total_pixels), str(defect_pixels),
            f'{defect_ratio:.8f}'
        ]
        row.extend(str(int(x)) for x in counts[:len(classes)])
        rows.append(row)
        row_records.append(
            dict(
                image_path=image_path,
                relative_path=str(relative_path),
                folder=folder,
                defect_ratio=defect_ratio))

    if not args.no_overlays and args.overlay_sample_per_folder > 0:
        overlay_rows = select_overlay_rows(row_records,
                                           args.overlay_sample_per_folder)
        overlay_manifest = overlay_dir / 'overlay_samples.csv'
        with overlay_manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['relative_path', 'folder', 'defect_ratio'])
            for item in overlay_rows:
                writer.writerow([
                    item['relative_path'], item['folder'],
                    f"{item['defect_ratio']:.8f}"
                ])

        for item in overlay_rows:
            image_path = item['image_path']
            relative_path = Path(item['relative_path'])
            result = inference_model(model, str(image_path))
            out_file = overlay_dir / relative_path.with_suffix('.png')
            mkdir_or_exist(out_file.parent)
            vis = show_result_pyplot(
                model,
                str(image_path),
                result,
                opacity=args.opacity,
                title=str(relative_path),
                draw_gt=False,
                draw_pred=True,
                show=False,
                with_labels=False)
            mmcv.imwrite(mmcv.rgb2bgr(vis), str(out_file))

    summary_path = out_dir / 'summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    folder_summary_path = out_dir / 'folder_summary.csv'
    folder_header = ['folder', 'num_images', 'total_pixels']
    folder_header.extend(f'pixels_{name}' for name in classes)
    folder_header.extend(f'ratio_{name}' for name in classes)
    with folder_summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(folder_header)
        for folder in sorted(folder_totals):
            counts = folder_totals[folder]
            total_pixels = int(counts.sum())
            ratios = counts / total_pixels if total_pixels else counts
            writer.writerow([
                folder,
                folder_images[folder],
                total_pixels,
                *[int(x) for x in counts],
                *[f'{x:.6f}' for x in ratios],
            ])

    if not args.no_overlays:
        print(f'Wrote overlays to {overlay_dir}')
    print(f'Wrote summary to {summary_path}')
    print(f'Wrote folder summary to {folder_summary_path}')


if __name__ == '__main__':
    main()

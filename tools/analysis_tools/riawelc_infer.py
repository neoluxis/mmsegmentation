# Copyright (c) OpenMMLab. All rights reserved.
"""Run cross-dataset inference on the RIAWELC classification-style folders.

The current RIAWELC copy has image folders such as ``Difetto1`` and
``NoDifetto`` but no pixel-level masks. This script saves qualitative overlays
and a CSV with predicted pixel counts per semantic class.
"""

from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist

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
    return parser.parse_args()


def iter_images(data_root: Path):
    for image_path in sorted(data_root.rglob('*')):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    overlay_dir = out_dir / 'overlays'
    mkdir_or_exist(overlay_dir)

    model = init_model(args.config, args.checkpoint, device=args.device)
    classes = model.dataset_meta['classes']
    header = ['relative_path', 'folder'] + [
        f'pixels_{name}' for name in classes
    ]
    rows = [','.join(header)]

    for image_path in iter_images(data_root):
        result = inference_model(model, str(image_path))
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
        counts = np.bincount(pred.reshape(-1), minlength=len(classes))
        relative_path = image_path.relative_to(data_root)
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

        row = [str(relative_path), relative_path.parts[0]]
        row.extend(str(int(x)) for x in counts[:len(classes)])
        rows.append(','.join(row))

    summary_path = out_dir / 'summary.csv'
    summary_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'Wrote overlays to {overlay_dir}')
    print(f'Wrote summary to {summary_path}')


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
"""Plot MMSegmentation scalar curves from ``vis_data/scalars.json``.

This helper targets MMEngine's local visualizer output. It accepts either a
training work directory, an indexed run directory, a timestamp run directory, a
``vis_data`` directory, or the scalar JSON file itself.
"""

import argparse
import csv
import json
import os
from pathlib import Path

if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import matplotlib.pyplot as plt


TRAIN_DEFAULTS = ('loss', 'decode.loss_ce', 'decode.acc_seg', 'lr')
VAL_DEFAULTS = ('mIoU', 'mAcc', 'aAcc')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'path',
        type=Path,
        help='work_dir, run dir, vis_data dir, or scalars.json path.')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Output directory. Defaults to <vis_data>/plots.')
    parser.add_argument(
        '--train-keys',
        nargs='+',
        default=list(TRAIN_DEFAULTS),
        help='Training scalar keys to plot against iter/step.')
    parser.add_argument(
        '--val-keys',
        nargs='+',
        default=list(VAL_DEFAULTS),
        help='Validation scalar keys to plot against iter/step.')
    parser.add_argument(
        '--dpi', type=int, default=160, help='Figure DPI. Default: 160.')
    return parser.parse_args()


def resolve_scalars_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path

    for direct in (path / 'scalars.json', path / 'vis_data' / 'scalars.json'):
        if direct.exists():
            return direct

    candidates = sorted(
        path.rglob('vis_data/scalars.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f'Cannot find scalars.json under {path}')


def read_json_lines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no} is not valid JSON') from exc
    return rows


def scalar_points(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in rows:
        if key not in row:
            continue
        step = row.get('step', row.get('iter'))
        if step is None:
            continue
        value = row[key]
        if isinstance(value, (int, float)):
            xs.append(int(step))
            ys.append(float(value))
    return xs, ys


def write_csv(rows: list[dict], out_file: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    with out_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plot(rows: list[dict], keys: list[str], out_file: Path, title: str,
              ylabel: str, dpi: int) -> bool:
    plotted = False
    plt.figure(figsize=(10, 5))
    for key in keys:
        xs, ys = scalar_points(rows, key)
        if not xs:
            continue
        marker = 'o' if len(xs) <= 40 else None
        linewidth = 1.8 if len(xs) <= 40 else 1.0
        plt.plot(xs, ys, label=key, marker=marker, linewidth=linewidth)
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close()
    return True


def main():
    args = parse_args()
    scalars_path = resolve_scalars_path(args.path)
    rows = read_json_lines(scalars_path)
    out_dir = args.out_dir or (scalars_path.parent / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_dir / 'scalars.csv')
    outputs = [out_dir / 'scalars.csv']

    if save_plot(rows, ['loss', 'decode.loss_ce'], out_dir / 'loss.png',
                 'Training Loss', 'loss', args.dpi):
        outputs.append(out_dir / 'loss.png')

    if save_plot(rows, ['lr', 'base_lr'], out_dir / 'learning_rate.png',
                 'Learning Rate', 'learning rate', args.dpi):
        outputs.append(out_dir / 'learning_rate.png')

    if save_plot(rows, ['decode.acc_seg'], out_dir / 'train_accuracy.png',
                 'Training Pixel Accuracy', 'accuracy (%)', args.dpi):
        outputs.append(out_dir / 'train_accuracy.png')

    val_keys = [key for key in args.val_keys if key in {k for r in rows for k in r}]
    if save_plot(rows, val_keys, out_dir / 'validation_metrics.png',
                 'Validation Metrics', 'score (%)', args.dpi):
        outputs.append(out_dir / 'validation_metrics.png')

    custom_train = [
        key for key in args.train_keys
        if key not in ('loss', 'decode.loss_ce', 'lr', 'base_lr',
                       'decode.acc_seg')
    ]
    if custom_train and save_plot(rows, custom_train,
                                  out_dir / 'custom_train_metrics.png',
                                  'Custom Training Metrics', 'value',
                                  args.dpi):
        outputs.append(out_dir / 'custom_train_metrics.png')

    print(f'Read scalars from {scalars_path}')
    for output in outputs:
        print(f'Wrote {output}')


if __name__ == '__main__':
    main()

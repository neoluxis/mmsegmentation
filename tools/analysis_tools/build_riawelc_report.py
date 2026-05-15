# Copyright (c) OpenMMLab. All rights reserved.
"""Build the selected RIAWELC cross-dataset visualization report."""

from argparse import ArgumentParser
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ZH = {
    'background': '背景',
    'air-hole': '气孔',
    'bite-edge': '咬边',
    'broken-arc': '断弧',
    'crack': '裂纹',
    'hollow-bead': '空心焊道',
    'overlap': '搭接/焊瘤',
    'slag-inclusion': '夹渣',
    'unfused': '未熔合',
}

EXPERIMENT_LABELS = {
    'segformer_mit-b2_8xb2-160k_swrd-768x768': 'SegFormer-B2 768 baseline',
    'segformer_mit-b2_8xb2-160k_swrd-768x768_clahe':
    'SegFormer-B2 768 CLAHE',
    'segformer_mit-b0_4xb2-40k_swrd-768x768_clahe':
    'SegFormer-B0 768 CLAHE',
    'pspnet_r50-d8_4xb2-40k_swrd-512x512_clahe': 'PSPNet R50 CLAHE',
    'mask2former_r50_4xb2-40k_swrd-512x512_clahe': 'Mask2Former R50 CLAHE',
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-root',
        type=Path,
        default=Path('work_dirs/riawelc_cross_dataset_testing_selected'))
    parser.add_argument(
        '--scalars-summary',
        type=Path,
        default=Path('work_dirs/analysis_outputs/scalars_summary.csv'))
    return parser.parse_args()


def experiment_label(exp: str) -> str:
    return EXPERIMENT_LABELS.get(exp, exp)


def class_name_from_col(col: str) -> str:
    return col.removeprefix('pixels_').removeprefix('ratio_')


def defect_pixel_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col for col in frame.columns
        if col.startswith('pixels_') and col != 'pixels_background'
    ]


def ratio_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col for col in frame.columns
        if col.startswith('ratio_') and col != 'ratio_background'
    ]


def summarize_models(folder_all: pd.DataFrame,
                     scalars: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    pixel_cols = defect_pixel_columns(folder_all)
    for exp, group in folder_all.groupby('experiment', sort=False):
        total = group['total_pixels'].sum()
        defect = group[pixel_cols].sum(axis=1).sum()
        no_defect = group[group['folder'].str.lower().eq('nodifetto')]
        no_total = no_defect['total_pixels'].sum()
        no_defect_pixels = no_defect[pixel_cols].sum(axis=1).sum()
        row = {
            'experiment': exp,
            'label': experiment_label(exp),
            'ria_defect_pixel_ratio': defect / total if total else np.nan,
            'ria_nodifetto_false_positive_ratio':
            no_defect_pixels / no_total if no_total else np.nan,
        }
        if scalars is not None and not scalars.empty:
            scalar_rows = scalars[scalars['exp'].eq(exp)].copy()
            if not scalar_rows.empty:
                scalar_rows = scalar_rows.sort_values(
                    'best_mIoU', ascending=False)
                row['swrd_best_mIoU'] = scalar_rows.iloc[0]['best_mIoU']
                row['swrd_best_step'] = scalar_rows.iloc[0]['best_step']
        rows.append(row)
    return pd.DataFrame(rows)


def save_bar(series: pd.Series, title: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind='bar', ax=ax, color='#2563eb')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_summary(model_summary: pd.DataFrame, report_dir: Path):
    indexed = model_summary.set_index('label')
    save_bar(indexed['ria_defect_pixel_ratio'],
             'RIAWELC predicted defect pixel ratio', 'ratio',
             report_dir / 'model_defect_pixel_ratio.png')
    save_bar(indexed['ria_nodifetto_false_positive_ratio'],
             'RIAWELC NoDifetto false positive pixel ratio', 'ratio',
             report_dir / 'nodifetto_false_positive_ratio.png')

    if {'swrd_best_mIoU', 'ria_nodifetto_false_positive_ratio'}.issubset(
            model_summary.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(model_summary['swrd_best_mIoU'],
                   model_summary['ria_nodifetto_false_positive_ratio'],
                   s=70,
                   color='#059669')
        for _, row in model_summary.iterrows():
            ax.annotate(row['label'],
                        (row['swrd_best_mIoU'],
                         row['ria_nodifetto_false_positive_ratio']),
                        fontsize=8,
                        xytext=(4, 4),
                        textcoords='offset points')
        ax.set_xlabel('SWRD best mIoU (%)')
        ax.set_ylabel('RIAWELC NoDifetto false positive ratio')
        ax.set_title('SWRD validation vs RIAWELC false positives')
        fig.tight_layout()
        fig.savefig(report_dir / 'swrd_miou_vs_riawelc_fp.png', dpi=180)
        plt.close(fig)


def plot_folder_stacks(folder_all: pd.DataFrame, report_dir: Path):
    ratio_cols = ratio_columns(folder_all)
    for exp, group in folder_all.groupby('experiment', sort=False):
        plot_data = group.set_index('folder')[ratio_cols].copy()
        plot_data.columns = [
            class_name_from_col(col) for col in plot_data.columns
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data.plot(kind='bar', stacked=True, ax=ax, width=0.75)
        ax.set_title(f'Predicted class ratios by folder: {experiment_label(exp)}')
        ax.set_ylabel('ratio over all pixels')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=0)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(report_dir / f'{exp}_folder_class_stack.png', dpi=180)
        plt.close(fig)


def overlay_paths(exp_dir: Path, max_images: int = 8) -> list[Path]:
    overlays = sorted((exp_dir / 'overlays').glob('*/*.png'))
    if len(overlays) <= max_images:
        return overlays
    indices = np.linspace(0, len(overlays) - 1, max_images, dtype=int)
    return [overlays[i] for i in indices]


def build_overlay_mosaics(out_root: Path, report_dir: Path,
                          experiments: list[str]) -> dict[str, Path | None]:
    mosaic_paths = {}
    for exp in experiments:
        images = overlay_paths(out_root / exp)
        if not images:
            mosaic_paths[exp] = None
            continue
        cols = 4
        rows = int(np.ceil(len(images) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, image_path in zip(axes.ravel(), images):
            ax.imshow(mpimg.imread(image_path))
            rel = image_path.relative_to(out_root / exp / 'overlays')
            ax.set_title(str(rel), fontsize=7)
            ax.axis('off')
        fig.suptitle(f'Overlay samples: {experiment_label(exp)}')
        fig.tight_layout()
        path = report_dir / f'{exp}_overlay_mosaic.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        mosaic_paths[exp] = path
    return mosaic_paths


def validate_outputs(out_root: Path, selected: pd.DataFrame,
                     image_all: pd.DataFrame, folder_all: pd.DataFrame):
    expected_images = 2443
    expected_folders = {'Difetto1', 'Difetto2', 'Difetto4', 'NoDifetto'}
    errors = []
    for exp in selected['experiment']:
        exp_images = image_all[image_all['experiment'].eq(exp)]
        exp_folders = folder_all[folder_all['experiment'].eq(exp)]
        if len(exp_images) != expected_images:
            errors.append(f'{exp}: summary rows {len(exp_images)} != 2443')
        folders = set(exp_folders['folder'])
        if folders != expected_folders:
            errors.append(f'{exp}: folders {sorted(folders)} != expected')
        mask_count = len(list((out_root / exp / 'masks').glob('*/*.png')))
        if mask_count != expected_images:
            errors.append(f'{exp}: masks {mask_count} != 2443')
        overlay_count = len(list((out_root / exp / 'overlays').glob('*/*.png')))
        if overlay_count < 1:
            errors.append(f'{exp}: no overlay samples found')
    if errors:
        raise RuntimeError('\n'.join(errors))


def write_report(out_root: Path, report_dir: Path, selected: pd.DataFrame,
                 model_summary: pd.DataFrame,
                 mosaic_paths: dict[str, Path | None]):
    rows = []
    for _, row in model_summary.iterrows():
        rows.append(
            '| {label} | {miou:.2f} | {defect:.4f} | {fp:.6f} |'.format(
                label=row['label'],
                miou=row.get('swrd_best_mIoU', np.nan),
                defect=row['ria_defect_pixel_ratio'],
                fp=row['ria_nodifetto_false_positive_ratio']))
    table = '\n'.join(rows)

    selected_rows = []
    for _, row in selected.iterrows():
        selected_rows.append(
            f"| `{row['experiment']}` | `{row['checkpoint_source']}` | "
            f"`{row['best_step']}` | `{row['checkpoint']}` |")

    overlay_lines = []
    for exp, path in mosaic_paths.items():
        if path is None:
            continue
        overlay_lines.append(
            f"### {experiment_label(exp)}\n\n"
            f"![{experiment_label(exp)}]({path.relative_to(report_dir)})\n")

    report = f"""# RIAWELC Cross-Dataset Validation Report

## 说明

RIAWELC 当前副本是按 `Difetto1`、`Difetto2`、`Difetto4`、`NoDifetto` 文件夹组织的分类式数据，没有像素级 mask。因此本报告是 **无标注定性 + 像素预测统计**，不能报告 RIAWELC mIoU。

本次仅使用 `RIAWELC/DB/testing`，共 2443 张图。每个入选模型保留全量 `summary.csv`、`folder_summary.csv` 和预测 mask；overlay 采用抽样保存，便于报告阅读和后续扩展。

## 入选模型与 checkpoint

| 实验 | checkpoint来源 | best_step | checkpoint |
|---|---|---:|---|
{chr(10).join(selected_rows)}

## 核心统计

| 模型 | SWRD best mIoU | RIAWELC defect pixel ratio | NoDifetto false positive ratio |
|---|---:|---:|---:|
{table}

## 图表

![model defect ratio](model_defect_pixel_ratio.png)

![nodifetto fp](nodifetto_false_positive_ratio.png)

![swrd miou fp](swrd_miou_vs_riawelc_fp.png)

各模型按 RIAWELC 文件夹的预测类别占比见：

{chr(10).join(f'- `{exp}_folder_class_stack.png`' for exp in selected['experiment'])}

## Overlay 抽样

{chr(10).join(overlay_lines)}

## 类别说明

| 英文类别 | 中文说明 |
|---|---|
{chr(10).join(f'| `{name}` | {zh} |' for name, zh in CLASS_ZH.items())}

## 数据保留位置

- 全量图片级统计：`{out_root / 'image_summary_all.csv'}`
- 全量文件夹级统计：`{out_root / 'folder_summary_all.csv'}`
- 入选模型清单：`{out_root / 'selected_models.csv'}`
- 每个模型预测 mask：`{out_root / '<experiment>' / 'masks'}`
- 每个模型抽样 overlay：`{out_root / '<experiment>' / 'overlays'}`
"""
    (report_dir / 'riawelc_cross_dataset_report.md').write_text(
        textwrap.dedent(report), encoding='utf-8')


def main():
    args = parse_args()
    out_root = args.out_root.resolve()
    report_dir = out_root / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(out_root / 'selected_models.csv')
    folder_all = pd.read_csv(out_root / 'folder_summary_all.csv')
    image_all = pd.read_csv(out_root / 'image_summary_all.csv')
    scalars = (pd.read_csv(args.scalars_summary)
               if args.scalars_summary.exists() else pd.DataFrame())

    validate_outputs(out_root, selected, image_all, folder_all)
    model_summary = summarize_models(folder_all, scalars)
    model_summary.to_csv(report_dir / 'model_summary.csv', index=False)

    plot_model_summary(model_summary, report_dir)
    plot_folder_stacks(folder_all, report_dir)
    mosaics = build_overlay_mosaics(out_root, report_dir,
                                    selected['experiment'].tolist())
    write_report(out_root, report_dir, selected, model_summary, mosaics)
    print(f'wrote {report_dir / "riawelc_cross_dataset_report.md"}')


if __name__ == '__main__':
    main()

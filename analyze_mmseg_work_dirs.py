from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("work_dirs")
OUT = ROOT / "analysis_outputs"

SWRD_CLASS_ALIASES = {
    "air-hole12(hollow-bead)-142": "air-hole",
    "air-hole7-028": "bite-edge",
    "broken-arc2-064": "broken-arc",
    "broken-arc2-156": "crack",
    "air-hole4(hollow-bead)-105": "hollow-bead",
    "broken-arc3(air-hole)-282": "overlap",
    "crack043": "slag-inclusion",
    "air-hole10-026": "unfused",
}


def parse_exp_from_name(exp: str) -> dict[str, str]:
    model = exp
    prep = "baseline"
    if exp.endswith("_clahe"):
        prep = "clahe"
        model = exp[:-6]
    elif exp.endswith("_gamma"):
        prep = "gamma"
        model = exp[:-6]
    elif exp.endswith("_gaussian"):
        prep = "gaussian"
        model = exp[:-9]

    arch = model
    for key in [
        "segformer_mit-b0",
        "segformer_mit-b2",
        "segformer_mit-b4",
        "deeplabv3plus_r50-d8",
        "mask2former_r50",
        "pspnet_r50-d8",
        "unet-s5-d16_fcn",
    ]:
        if model.startswith(key):
            arch = key
            break

    return {
        "exp": exp,
        "model": model,
        "arch": arch,
        "prep": prep,
        "size": "768" if "768x768" in exp else "512",
        "budget": "160k" if "160k" in exp else "40k",
    }


def parse_exp(path: Path) -> dict[str, str]:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    meta = parse_exp_from_name(parts[0])
    meta["run"] = ""
    meta["timestamp"] = ""
    if len(parts) >= 4 and parts[-2] == "vis_data":
        meta["timestamp"] = parts[-3]
        meta["run"] = parts[-4] if parts[-4].startswith("run_") else ""
    return meta


def collect_scalars() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    curves: list[dict] = []

    for path in sorted(ROOT.glob("**/vis_data/scalars.json")):
        path_text = str(path)
        if "analysis_outputs" in path_text or "riawelc_cross_dataset" in path_text:
            continue

        meta = parse_exp(path)
        vals = []
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mIoU" not in data:
                continue

            vals.append(data)
            curves.append(
                {
                    **meta,
                    **{
                        key: data.get(key)
                        for key in [
                            "step",
                            "aAcc",
                            "mIoU",
                            "mAcc",
                            "mDice",
                            "mFscore",
                            "mPrecision",
                            "mRecall",
                            "data_time",
                            "time",
                        ]
                    },
                    "scalars_path": str(path),
                }
            )

        if not vals:
            continue

        best = max(vals, key=lambda item: item.get("mIoU", -1))
        last = vals[-1]
        rows.append(
            {
                **meta,
                "num_val_points": len(vals),
                "best_step": best.get("step"),
                "best_mIoU": best.get("mIoU"),
                "best_aAcc": best.get("aAcc"),
                "best_mAcc": best.get("mAcc"),
                "best_mDice": best.get("mDice"),
                "best_mFscore": best.get("mFscore"),
                "best_mPrecision": best.get("mPrecision"),
                "best_mRecall": best.get("mRecall"),
                "last_step": last.get("step"),
                "last_mIoU": last.get("mIoU"),
                "last_aAcc": last.get("aAcc"),
                "scalars_path": str(path),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["best_mIoU", "best_aAcc"], ascending=False
    )
    curves_df = pd.DataFrame(curves)
    return summary, curves_df


def parse_log(path: Path) -> dict | None:
    lines = path.read_text(errors="ignore").splitlines()
    candidates = []
    for index, line in enumerate(lines):
        match = re.search(
            r"Iter\(val\).*?aAcc:\s*([0-9.]+)\s+mIoU:\s*([0-9.]+)\s+mAcc:\s*([0-9.]+)",
            line,
        )
        if not match:
            continue

        start = None
        for table_index in range(index - 1, max(-1, index - 45), -1):
            if "per class results" in lines[table_index]:
                start = table_index
                break

        table = []
        if start is not None:
            for table_line in lines[start:index]:
                if (
                    "|" not in table_line
                    or "Class" in table_line
                    or "+---" in table_line
                ):
                    continue
                row = table_line[table_line.find("|") :]
                parts = [part.strip() for part in row.strip("|").split("|")]
                if len(parts) < 3:
                    continue
                try:
                    table.append(
                        {
                            "Class": SWRD_CLASS_ALIASES.get(parts[0], parts[0]),
                            "IoU": float(parts[1]),
                            "Acc": float(parts[2]),
                            "Dice": float(parts[3]) if len(parts) > 3 else None,
                            "Fscore": float(parts[4]) if len(parts) > 4 else None,
                            "Precision": float(parts[5]) if len(parts) > 5 else None,
                            "Recall": float(parts[6]) if len(parts) > 6 else None,
                        }
                    )
                except ValueError:
                    continue

        candidates.append(
            {
                "aAcc": float(match.group(1)),
                "mIoU": float(match.group(2)),
                "mAcc": float(match.group(3)),
                "table": table,
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["mIoU"])


def collect_per_class() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob("**/*.log")):
        path_text = str(path)
        if "analysis_outputs" in path_text or "riawelc_cross_dataset" in path_text:
            continue

        parsed = parse_log(path)
        if not parsed:
            continue

        rel = path.relative_to(ROOT)
        meta = parse_exp_from_name(rel.parts[0])
        meta["run"] = next((part for part in rel.parts if part.startswith("run_")), "")
        meta["timestamp"] = path.stem
        for row in parsed["table"]:
            rows.append(
                {
                    **meta,
                    "best_log_mIoU": parsed["mIoU"],
                    "best_log_aAcc": parsed["aAcc"],
                    "best_log_mAcc": parsed["mAcc"],
                    "log_path": str(path),
                    **row,
                }
            )
    return pd.DataFrame(rows)


def collect_riawelc() -> pd.DataFrame:
    path = ROOT / "riawelc_cross_dataset" / "folder_summary_all.csv"
    if path.exists():
        raw = pd.read_csv(path)
    else:
        frames = []
        for summary_path in sorted((ROOT / "riawelc_cross_dataset").glob("*/folder_summary.csv")):
            frame = pd.read_csv(summary_path)
            frame.insert(0, "experiment", summary_path.parent.name)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)

    if raw.empty:
        return pd.DataFrame()
    ratio_cols = [
        col for col in raw.columns if col.startswith("ratio_") and col != "ratio_background"
    ]
    rows = []
    for exp, group in raw.groupby("experiment"):
        meta = parse_exp_from_name(exp)
        total = group["total_pixels"].sum()
        pixel_cols = [col.replace("ratio_", "pixels_") for col in ratio_cols]
        defect_pixels = sum(group[col].sum() for col in pixel_cols if col in group.columns)

        no_defect = group[group["folder"].str.lower().eq("nodifetto")]
        no_defect_total = no_defect["total_pixels"].sum() if len(no_defect) else 0
        no_defect_pred = (
            sum(no_defect[col].sum() for col in pixel_cols if col in no_defect.columns)
            if len(no_defect)
            else 0
        )

        rows.append(
            {
                **meta,
                "ria_defect_pixel_ratio": defect_pixels / total if total else None,
                "ria_nodifetto_false_positive_ratio": (
                    no_defect_pred / no_defect_total if no_defect_total else None
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("ria_nodifetto_false_positive_ratio")


def build_segformer_preprocess_compare(summary: pd.DataFrame) -> pd.DataFrame:
    segformer = summary[
        summary["arch"].isin(["segformer_mit-b0", "segformer_mit-b2"])
    ].copy()
    if segformer.empty:
        return pd.DataFrame()

    pivot = segformer.pivot_table(
        index=["arch", "size", "budget"],
        columns="prep",
        values="best_mIoU",
        aggfunc="max",
    ).reset_index()

    prep_cols = [
        col for col in ["baseline", "clahe", "gamma", "gaussian"] if col in pivot.columns
    ]
    ordered_cols = ["arch", "size", "budget", *prep_cols]
    compare = pivot[ordered_cols].copy()
    if {"baseline", "clahe"}.issubset(compare.columns):
        compare = compare.assign(
            clahe_minus_baseline=compare["clahe"] - compare["baseline"]
        )
    if prep_cols:
        compare = compare.assign(
            best_prep=compare[prep_cols].idxmax(axis=1),
            best_mIoU=compare[prep_cols].max(axis=1),
        )
    return compare.sort_values(["arch", "budget", "size"])


def build_preprocess_compare(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["arch", "size", "budget"],
        columns="prep",
        values="best_mIoU",
        aggfunc="max",
    ).reset_index()

    prep_cols = [
        col for col in ["baseline", "clahe", "gamma", "gaussian"] if col in pivot.columns
    ]
    compare = pivot[["arch", "size", "budget", *prep_cols]].copy()
    compare = compare[compare[prep_cols].notna().sum(axis=1).gt(1)]
    if {"baseline", "clahe"}.issubset(compare.columns):
        compare = compare.assign(
            clahe_minus_baseline=compare["clahe"] - compare["baseline"]
        )
    if prep_cols:
        compare = compare.assign(
            best_prep=compare[prep_cols].idxmax(axis=1),
            best_mIoU=compare[prep_cols].max(axis=1),
        )
    return compare.sort_values(["budget", "size", "arch"])


def save_plots(summary: pd.DataFrame, curves: pd.DataFrame, per_class: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    top = summary.head(25).copy()
    top = top.assign(
        label=(
            top["arch"].str.replace("segformer_", "sf_", regex=False)
            + " / "
            + top["prep"]
            + " / "
            + top["size"]
            + " / "
            + top["budget"]
            + " / "
            + top["run"].fillna("")
        )
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["label"][::-1], top["best_mIoU"][::-1], color="#2563eb")
    ax.axvline(80, color="#dc2626", linestyle="--", linewidth=1, label="target 80")
    ax.set_xlabel("Best mIoU (%)")
    ax.set_title("SWRD validation best mIoU by run")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "best_miou_top25.png", dpi=180)
    plt.close(fig)

    preprocess_compare = build_preprocess_compare(summary)
    if len(preprocess_compare):
        prep_cols = [
            col
            for col in ["baseline", "clahe", "gamma", "gaussian"]
            if col in preprocess_compare.columns
        ]
        plot_data = preprocess_compare.set_index(
            ["arch", "size", "budget"]
        )[prep_cols].copy()
        plot_data.index = [
            f"{arch.replace('segformer_mit-', 'SF-')} / {size} / {budget}"
            for arch, size, budget in plot_data.index
        ]
        height = max(6, min(12, 0.45 * len(plot_data) + 3))
        fig, ax = plt.subplots(figsize=(14, height))
        plot_data.plot(kind="bar", ax=ax)
        ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_xlabel("Experiment group")
        ax.set_ylabel("Best mIoU (%)")
        ax.set_title("Preprocessing ablation across experiment groups")
        ax.legend(title="preprocess")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(OUT / "preprocess_ablation_all.png", dpi=180)
        plt.close(fig)

    segformer = curves[curves["arch"].astype(str).str.contains("segformer", na=False)]
    if len(segformer):
        fig, ax = plt.subplots(figsize=(11, 6))
        for keys, group in segformer.groupby(
            ["arch", "prep", "size", "budget", "exp", "run", "timestamp"]
        ):
            arch, prep, size, budget, _exp, run, timestamp = keys
            include = (
                (arch == "segformer_mit-b0" and size == "512" and budget == "40k")
                or size == "768"
                or arch == "segformer_mit-b2"
            )
            if not include:
                continue
            label = f'{arch.replace("segformer_", "")} {prep} {size} {budget} {run or timestamp}'
            ax.plot(
                group["step"],
                group["mIoU"],
                marker="o",
                linewidth=1.3,
                markersize=3,
                label=label,
            )
        ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_xlabel("Iter")
        ax.set_ylabel("mIoU (%)")
        ax.set_title("SegFormer validation curves")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "segformer_validation_curves.png", dpi=180)
        plt.close(fig)

        b2_160k = segformer[
            (segformer["arch"].eq("segformer_mit-b2"))
            & (segformer["budget"].eq("160k"))
        ]
        if len(b2_160k):
            fig, ax = plt.subplots(figsize=(11, 6))
            for keys, group in b2_160k.groupby(["prep", "size", "exp", "run", "timestamp"]):
                prep, size, _exp, run, timestamp = keys
                label = f"B2 {prep} {size} {run or timestamp}"
                ax.plot(
                    group["step"],
                    group["mIoU"],
                    marker="o",
                    linewidth=1.5,
                    markersize=3,
                    label=label,
                )
            ax.axhline(80, color="#dc2626", linestyle="--", linewidth=1)
            ax.set_xlabel("Iter")
            ax.set_ylabel("mIoU (%)")
            ax.set_title("SegFormer-B2 160k validation curves")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(OUT / "segformer_b2_160k_validation_curves.png", dpi=180)
            plt.close(fig)

    if len(per_class) and len(summary):
        best_exp = summary.iloc[0]["exp"]
        current = per_class[per_class["exp"].eq(best_exp)].copy()
        if len(current):
            best_log_miou = current["best_log_mIoU"].max()
            current = current[current["best_log_mIoU"].eq(best_log_miou)].sort_values("IoU")
            fig, ax = plt.subplots(figsize=(11, 5.5))
            colors = ["#dc2626" if value < 70 else "#059669" for value in current["IoU"]]
            ax.barh(current["Class"], current["IoU"], color=colors)
            ax.axvline(
                70,
                color="#dc2626",
                linestyle="--",
                linewidth=1,
                label="per-class target 70",
            )
            ax.set_xlabel("IoU (%)")
            ax.set_title(f"Per-class IoU: {best_exp}")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(OUT / "best_model_per_class_iou.png", dpi=180)
            plt.close(fig)


def save_report(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    riawelc: pd.DataFrame,
) -> None:
    cols = [
        "arch",
        "prep",
        "size",
        "budget",
        "run",
        "timestamp",
        "best_step",
        "best_mIoU",
        "best_aAcc",
        "best_mAcc",
        "best_mDice",
        "last_mIoU",
        "scalars_path",
    ]
    report = ["# mmsegmentation/work_dirs 指标分析摘要\n"]
    report.append("## 1. Top runs by best mIoU\n")
    report.append(summary[cols].head(20).to_markdown(index=False, floatfmt=".2f"))

    base = summary[(summary["size"] == "512") & (summary["budget"] == "40k")]
    report.append("\n\n## 2. 512x512 / 40k 模型与预处理对比\n")
    report.append(
        base.sort_values("best_mIoU", ascending=False)[cols]
        .head(30)
        .to_markdown(index=False, floatfmt=".2f")
    )

    segformer_preprocess = build_segformer_preprocess_compare(summary)
    if len(segformer_preprocess):
        report.append("\n\n## 3. SegFormer-B0/B2 预处理对比\n")
        report.append(
            segformer_preprocess.to_markdown(index=False, floatfmt=".2f")
        )

    b2_160k = summary[
        (summary["arch"].eq("segformer_mit-b2")) & (summary["budget"].eq("160k"))
    ]
    if len(b2_160k):
        report.append("\n\n## 4. SegFormer-B2 / 160k 训练结果\n")
        report.append(
            b2_160k.sort_values(["best_mIoU", "best_aAcc"], ascending=False)[cols]
            .to_markdown(index=False, floatfmt=".2f")
        )

        pivot = b2_160k.pivot_table(
            index="size", columns="prep", values="best_mIoU", aggfunc="max"
        ).sort_index()
        prep_cols = [
            col for col in ["baseline", "clahe", "gamma", "gaussian"] if col in pivot
        ]
        compare = pivot[prep_cols].copy()
        if {"baseline", "clahe"}.issubset(compare.columns):
            compare = compare.assign(
                clahe_minus_baseline=compare["clahe"] - compare["baseline"]
            )
        report.append("\n\n## 5. SegFormer-B2 / 160k 预处理对比\n")
        report.append(compare.reset_index().to_markdown(index=False, floatfmt=".2f"))

    if len(per_class):
        best_exp = summary.iloc[0]["exp"]
        best_log_miou = per_class[per_class["exp"].eq(best_exp)]["best_log_mIoU"].max()
        current = per_class[
            (per_class["exp"].eq(best_exp))
            & (per_class["best_log_mIoU"].eq(best_log_miou))
        ][["Class", "IoU", "Acc", "Dice", "Precision", "Recall"]]
        report.append(f"\n\n## 6. 最优实验 `{best_exp}` 的逐类指标\n")
        report.append(current.to_markdown(index=False, floatfmt=".2f"))

    if len(riawelc):
        report.append("\n\n## 7. RIAWELC 跨域推理摘要\n")
        report.append(
            riawelc[
                [
                    "arch",
                    "prep",
                    "size",
                    "budget",
                    "ria_defect_pixel_ratio",
                    "ria_nodifetto_false_positive_ratio",
                ]
            ]
            .head(30)
            .to_markdown(index=False, floatfmt=".6f")
        )

    (OUT / "analysis_report.md").write_text("\n".join(report))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summary, curves = collect_scalars()
    per_class = collect_per_class()
    riawelc = collect_riawelc()

    summary.to_csv(OUT / "scalars_summary.csv", index=False)
    curves.to_csv(OUT / "validation_curves.csv", index=False)
    if len(per_class):
        per_class.to_csv(OUT / "per_class_best.csv", index=False)
    if len(riawelc):
        riawelc.to_csv(OUT / "riawelc_summary.csv", index=False)

    save_plots(summary, curves, per_class)
    save_report(summary, per_class, riawelc)

    cols = [
        "arch",
        "prep",
        "size",
        "budget",
        "run",
        "timestamp",
        "best_step",
        "best_mIoU",
        "best_aAcc",
        "best_mAcc",
        "last_mIoU",
    ]
    print(f"Wrote outputs to {OUT}")
    print(summary[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

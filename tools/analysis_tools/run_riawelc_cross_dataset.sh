#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMSEG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MMSEG_ROOT}"

DATA_ROOT="${DATA_ROOT:-../datasets/RIAWELC/DB/testing}"
OUT_ROOT="${OUT_ROOT:-work_dirs/riawelc_cross_dataset_testing_selected}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OVERLAY_SAMPLE_PER_FOLDER="${OVERLAY_SAMPLE_PER_FOLDER:-20}"
WRITE_MASKS="${WRITE_MASKS:-1}"
SELECTED_EXPERIMENTS="${SELECTED_EXPERIMENTS:-segformer_mit-b2_8xb2-160k_swrd-768x768 segformer_mit-b2_8xb2-160k_swrd-768x768_clahe segformer_mit-b0_4xb2-40k_swrd-768x768_clahe pspnet_r50-d8_4xb2-40k_swrd-512x512_clahe mask2former_r50_4xb2-40k_swrd-512x512_clahe}"
SCALARS_SUMMARY="${SCALARS_SUMMARY:-work_dirs/analysis_outputs/scalars_summary.csv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mmseg}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

count_testing_images() {
  find "${DATA_ROOT}" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
    -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
}

best_step_for() {
  local name="$1"
  "${PYTHON_BIN}" -c '
import csv
import os
from pathlib import Path

name = os.environ["EXP_NAME"]
path = Path(os.environ["SCALARS_SUMMARY"])
if not path.exists():
    raise SystemExit(0)
with path.open(newline="", encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row.get("exp") == name]
if not rows:
    raise SystemExit(0)
rows.sort(key=lambda row: float(row.get("best_mIoU") or -1), reverse=True)
step = rows[0].get("best_step", "")
print(str(int(float(step))) if step else "")
' 2>/dev/null || true
}

find_checkpoint() {
  local name="$1"
  local best_step="$2"
  local work_dir="work_dirs/${name}"
  local checkpoint=""
  local marker=""

  if [[ -n "${best_step}" ]]; then
    checkpoint="$(find "${work_dir}" -name "iter_${best_step}.pth" -type f -print 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${checkpoint}" ]]; then
      echo "${checkpoint}|best_step"
      return
    fi
  fi

  checkpoint="$(find "${work_dir}" -name 'best_mIoU*.pth' -type f -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}|best_mIoU"
    return
  fi

  marker="$(find "${work_dir}" -name last_checkpoint -type f -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${marker}" ]]; then
    checkpoint="$(tr -d '\r\n' < "${marker}")"
    if [[ -n "${checkpoint}" && -f "${checkpoint}" ]]; then
      echo "${checkpoint}|last_checkpoint"
      return
    fi
  fi

  checkpoint="$(find "${work_dir}" -name 'iter_*.pth' -type f -print 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}|latest_iter"
  fi
}

image_count="$(count_testing_images | tr -d ' ')"
if [[ "${image_count}" != "2443" ]]; then
  echo "RIAWELC testing image count ${image_count} != expected 2443" >&2
  exit 1
fi

selected_csv="${OUT_ROOT}/selected_models.csv"
echo "experiment,config,checkpoint,checkpoint_source,best_step" > "${selected_csv}"

for name in ${SELECTED_EXPERIMENTS}; do
  config="configs/swrd/${name}.py"
  if [[ ! -f "${config}" ]]; then
    echo "missing config: ${config}" >&2
    exit 1
  fi

  best_step="$(EXP_NAME="${name}" SCALARS_SUMMARY="${SCALARS_SUMMARY}" best_step_for "${name}")"
  checkpoint_info="$(find_checkpoint "${name}" "${best_step}")"
  if [[ -z "${checkpoint_info}" ]]; then
    echo "missing checkpoint for ${name}" >&2
    exit 1
  fi
  checkpoint="${checkpoint_info%%|*}"
  checkpoint_source="${checkpoint_info##*|}"

  echo "${name},${config},${checkpoint},${checkpoint_source},${best_step}" >> "${selected_csv}"
  echo "run ${name}: ${checkpoint} (${checkpoint_source}, best_step=${best_step:-n/a})"

  out_dir="${OUT_ROOT}/${name}"
  mask_args=()
  if [[ "${WRITE_MASKS}" == "1" ]]; then
    mask_args+=(--write-masks)
  fi

  "${PYTHON_BIN}" tools/analysis_tools/riawelc_infer.py \
    "${config}" \
    "${checkpoint}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE}" \
    --overlay-sample-per-folder "${OVERLAY_SAMPLE_PER_FOLDER}" \
    "${mask_args[@]}"
done

OUT_ROOT="${OUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import pandas as pd

out_root = Path(os.environ["OUT_ROOT"])
folder_frames = []
image_frames = []
for path in sorted(out_root.glob("*/folder_summary.csv")):
    frame = pd.read_csv(path)
    frame.insert(0, "experiment", path.parent.name)
    folder_frames.append(frame)
for path in sorted(out_root.glob("*/summary.csv")):
    frame = pd.read_csv(path)
    frame.insert(0, "experiment", path.parent.name)
    image_frames.append(frame)
if folder_frames:
    pd.concat(folder_frames, ignore_index=True).to_csv(
        out_root / "folder_summary_all.csv", index=False)
    print(f"wrote {out_root / 'folder_summary_all.csv'}")
if image_frames:
    pd.concat(image_frames, ignore_index=True).to_csv(
        out_root / "image_summary_all.csv", index=False)
    print(f"wrote {out_root / 'image_summary_all.csv'}")
PY

"${PYTHON_BIN}" tools/analysis_tools/build_riawelc_report.py \
  --out-root "${OUT_ROOT}" \
  --scalars-summary "${SCALARS_SUMMARY}"

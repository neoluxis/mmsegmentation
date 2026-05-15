#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../datasets/RIAWELC/DB/testing}"
OUT_ROOT="${OUT_ROOT:-work_dirs/riawelc_cross_dataset}"
CONFIG_GLOB="${CONFIG_GLOB:-configs/swrd/segformer_mit-b2_8xb2-160k_swrd-*.py}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-}"
DEVICE="${DEVICE:-cuda:0}"
WRITE_OVERLAYS="${WRITE_OVERLAYS:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mmseg}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

infer_iter() {
  local name="$1"
  if [[ -n "${CHECKPOINT_ITER}" ]]; then
    echo "${CHECKPOINT_ITER}"
  elif [[ "${name}" =~ _([0-9]+)k_ ]]; then
    echo "$((BASH_REMATCH[1] * 1000))"
  else
    echo "40000"
  fi
}

find_checkpoint() {
  local name="$1"
  local iter="$2"
  local work_dir="work_dirs/${name}"
  local marker=""
  local checkpoint=""

  marker="$(find "${work_dir}" -name last_checkpoint -type f -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${marker}" ]]; then
    checkpoint="$(tr -d '\r\n' < "${marker}")"
    if [[ -n "${checkpoint}" && -f "${checkpoint}" ]]; then
      echo "${checkpoint}"
      return
    fi
  fi

  checkpoint="$(find "${work_dir}" -name 'best_mIoU*.pth' -type f -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}"
    return
  fi

  find "${work_dir}" -name "iter_${iter}.pth" -type f -print 2>/dev/null | sort | tail -n 1 || true
}

configs=( ${CONFIG_GLOB} )
if [[ ${#configs[@]} -eq 0 || ! -e "${configs[0]}" ]]; then
  echo "no configs matched: ${CONFIG_GLOB}"
  exit 1
fi

for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  iter="$(infer_iter "${name}")"
  checkpoint="$(find_checkpoint "${name}" "${iter}")"

  if [[ -z "${checkpoint}" ]]; then
    echo "skip ${name}: checkpoint not found (looked for last_checkpoint, best_mIoU*.pth, iter_${iter}.pth)"
    continue
  fi

  out_dir="${OUT_ROOT}/${name}"
  echo "run ${name} with ${checkpoint}"
  if [[ "${WRITE_OVERLAYS}" == "1" ]]; then
    python tools/analysis_tools/riawelc_infer.py \
      "${config}" \
      "${checkpoint}" \
      --data-root "${DATA_ROOT}" \
      --out-dir "${out_dir}" \
      --device "${DEVICE}"
  else
    python tools/analysis_tools/riawelc_infer.py \
      "${config}" \
      "${checkpoint}" \
      --data-root "${DATA_ROOT}" \
      --out-dir "${out_dir}" \
      --device "${DEVICE}" \
      --no-overlays
  fi
done

OUT_ROOT="${OUT_ROOT}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

out_root = Path(os.environ["OUT_ROOT"])
frames = []
for path in sorted(out_root.glob("*/folder_summary.csv")):
    frame = pd.read_csv(path)
    frame.insert(0, "experiment", path.parent.name)
    frames.append(frame)
if frames:
    pd.concat(frames, ignore_index=True).to_csv(out_root / "folder_summary_all.csv", index=False)
    print(f"wrote {out_root / 'folder_summary_all.csv'}")
PY

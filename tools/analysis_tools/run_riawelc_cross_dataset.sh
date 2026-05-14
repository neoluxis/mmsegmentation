#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../datasets/RIAWELC/DB/testing}"
OUT_ROOT="${OUT_ROOT:-work_dirs/riawelc_cross_dataset}"
DEVICE="${DEVICE:-cuda:0}"
WRITE_OVERLAYS="${WRITE_OVERLAYS:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mmseg}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

for config in configs/swrd/*40k_swrd-512x512*.py; do
  name="$(basename "${config}" .py)"
  checkpoint="$(find "work_dirs/${name}" -path '*/iter_40000.pth' -print | sort | tail -n 1)"

  if [[ -z "${checkpoint}" ]]; then
    echo "skip ${name}: iter_40000.pth not found"
    continue
  fi

  out_dir="${OUT_ROOT}/${name}"
  echo "run ${name}"
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

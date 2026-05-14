#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-work_dirs/swrd_val_visualizations}"
CONFIG_GLOB="${CONFIG_GLOB:-configs/swrd/*40k_swrd-512x512*.py}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mmseg}"

mkdir -p "${OUT_ROOT}" "${MPLCONFIGDIR}"

configs=( ${CONFIG_GLOB} )
if [[ ${#configs[@]} -eq 0 || ! -e "${configs[0]}" ]]; then
  echo "no configs matched: ${CONFIG_GLOB}"
  exit 1
fi

for config in "${configs[@]}"; do
  name="$(basename "${config}" .py)"
  checkpoint="$(find "work_dirs/${name}" -path '*/iter_40000.pth' -print | sort | tail -n 1)"

  if [[ -z "${checkpoint}" ]]; then
    echo "skip ${name}: iter_40000.pth not found"
    continue
  fi

  out_dir="${OUT_ROOT}/${name}"
  echo "visualize ${name}"
  python tools/test.py \
    "${config}" \
    "${checkpoint}" \
    --show-dir "${out_dir}" \
    --work-dir "${out_dir}/eval"
done

#!/bin/bash
pwd; hostname; date

source ./scripts/e1/config.sh

mkdir -p "${data_dir}" "${tmp_dir}"

apptainer_run_params=(
    --bind ${data_dir}:/data
    --bind ${tmp_dir}:/tmp
    container.sif
    python3 ./src/e2.py
    --data-path /data
)

if [ ${slurm:-0} -eq 1 ]; then
  echo "[*] SLURM"
  apptainer run "${apptainer_run_params[@]}"
else
  echo "[*] NO SLURM"
  apptainer run --no-home "${apptainer_run_params[@]}"
fi

date

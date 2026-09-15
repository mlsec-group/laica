###
### Runs one search cell or worker, selected by the array index
###

mkdir -p "${data_dir}" "${tmp_dir}" "${cache_dir}"

gpu_params=()
if [ ${use_gpu:-0} -eq 1 ]; then
    gpu_params=(--nv)
fi

apptainer_run_params=(
    "${gpu_params[@]}"
    --bind ${data_dir}:/data
    --bind ${tmp_dir}:/tmp
    --bind ${cache_dir}:/cache
    container.sif
    "${search_command[@]}"
)

export APPTAINERENV_HF_HOME=/cache

if [ ${slurm:-0} -eq 1 ]; then
  echo "[*] SLURM"
  apptainer run "${apptainer_run_params[@]}"
else
  echo "[*] NO SLURM"
  APPTAINERENV_CUDA_VISIBLE_DEVICES=${cuda_visible_devices} apptainer run --no-home "${apptainer_run_params[@]}"
fi

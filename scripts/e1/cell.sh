###
### Runs a single cell of the E1 grid, selected by the array index
###

index=${SLURM_ARRAY_TASK_ID:-0}

seed=$(( index % 10 ))
index=$(( index / 10 ))
variant=${variants[$(( index % 2 ))]}
index=$(( index / 2 ))
fraction=${fractions[${index}]}

if [ "${variant}" = "augmented" ]; then
    learning_rate=${learning_rate_augmented[${fraction}]}
    n_augmentations=${num_augmentations[${fraction}]}
    variant_params=("${augmentation_params[@]}")
else
    learning_rate=${learning_rate_baseline[${fraction}]}
    n_augmentations=0
    variant_params=()
fi

eval_params=()
if [ -n "${max_eval_samples}" ]; then
    eval_params=(--max-eval-samples "${max_eval_samples}")
fi

search_params=()
if [ ${use_best_params:-0} -eq 1 ]; then
    search_params=(
        --transformations-from-search /data/search
        --training-params-from-search /data/search
    )
fi

gpu_params=()
if [ ${use_gpu:-0} -eq 1 ]; then
    gpu_params=(--nv)
fi

mkdir -p "${data_dir}" "${tmp_dir}" "${cache_dir}"

echo "[*] ${task} ${variant} 1/${fraction} seed ${seed} lr ${learning_rate} num-augmentations ${n_augmentations}"

apptainer_run_params=(
    "${gpu_params[@]}"
    --bind ${data_dir}:/data
    --bind ${tmp_dir}:/tmp
    --bind ${cache_dir}:/cache
    container.sif
    python3 ./src/e1.py
    --task "${task}"
    --fraction "${fraction}"
    --seed "${seed}"
    --data-path /data
    --cache-dir /cache
    --nprocs "${nprocs}"
    --learning-rate "${learning_rate}"
    --num-augmentations "${n_augmentations}"
    "${task_params[@]}"
    "${variant_params[@]}"
    "${eval_params[@]}"
    "${search_params[@]}"
)

export APPTAINERENV_HF_HOME=/cache

if [ ${slurm:-0} -eq 1 ]; then
  echo "[*] SLURM"
  apptainer run "${apptainer_run_params[@]}"
else
  echo "[*] NO SLURM"
  APPTAINERENV_CUDA_VISIBLE_DEVICES=${cuda_visible_devices} apptainer run --no-home "${apptainer_run_params[@]}"
fi

#!/bin/bash
#SBATCH --job-name=laica-search-runs
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=data/log/laica-search-runs/array_%A-%a.out
#SBATCH --array=0-89
pwd; hostname; date

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 task"
    exit 1
fi

task=$1

source ./scripts/e1/search/config.sh
source ./scripts/e1/params/${task}.sh

index=${SLURM_ARRAY_TASK_ID:-0}

learning_rate=${grid_learning_rates[$(( index % 3 ))]}
index=$(( index / 3 ))
n_augmentations=${grid_augmentation_runs[$(( index % 3 ))]}
index=$(( index / 3 ))
fraction=${fractions[${index}]}

transformation_params=("${augmentation_params[@]}")
if [ ${use_best_params:-0} -eq 1 ]; then
    transformation_params=(--transformations-from-search /data/search)
fi

echo "[*] ${task} runs grid 1/${fraction} lr ${learning_rate} num-augmentations ${n_augmentations}"

search_command=(
    python3 ./src/e1.py
    --task "${task}"
    --fraction "${fraction}"
    --seed "${search_seed}"
    --data-path /data
    --cache-dir /cache
    --nprocs "${nprocs}"
    --learning-rate "${learning_rate}"
    --num-augmentations "${n_augmentations}"
    --output-dir "/data/search/runs/${task}/${fraction}/${learning_rate}-${n_augmentations}"
    --skip-test
    "${transformation_params[@]}"
)

source ./scripts/e1/search/run.sh

date

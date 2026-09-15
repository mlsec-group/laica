#!/bin/bash
#SBATCH --job-name=laica-search-transformations
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=data/log/laica-search-transformations/array_%A-%a.out
#SBATCH --array=0-39
pwd; hostname; date

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 task"
    exit 1
fi

task=$1

source ./scripts/e1/search/config.sh
source ./scripts/e1/params/${task}.sh

echo "[*] ${task} transformation search worker ${SLURM_ARRAY_TASK_ID:-0}, ${trials_per_worker} trials"

search_command=(
    python3 ./src/search_transformations.py
    --task "${task}"
    --data-path /data
    --cache-dir /cache
    --nprocs "${nprocs}"
    --trials "${trials_per_worker}"
    --seed "${search_seed}"
    --fraction "${search_fraction}"
    --learning-rate "${search_learning_rate}"
    --num-augmentations "${search_num_augmentations}"
    "${search_task_params[@]}"
)

source ./scripts/e1/search/run.sh

date

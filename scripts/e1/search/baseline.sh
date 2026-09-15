#!/bin/bash
#SBATCH --job-name=laica-search-baseline
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=data/log/laica-search-baseline/array_%A-%a.out
#SBATCH --array=0-29
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
fraction=${fractions[$(( index / 3 ))]}

echo "[*] ${task} baseline grid 1/${fraction} lr ${learning_rate}"

search_command=(
    python3 ./src/e1.py
    --task "${task}"
    --fraction "${fraction}"
    --seed "${search_seed}"
    --data-path /data
    --cache-dir /cache
    --nprocs "${nprocs}"
    --learning-rate "${learning_rate}"
    --num-augmentations 0
    --output-dir "/data/search/baseline/${task}/${fraction}/${learning_rate}"
    --skip-test
)

source ./scripts/e1/search/run.sh

date

#!/bin/bash
#SBATCH --job-name=laica-e3-function_similarity
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=data/log/laica-e3-function_similarity/array_%A-%a.out
#SBATCH --array=0-199
pwd; hostname; date

source ./scripts/e3/config.sh

task=function_similarity

source ./scripts/e3/params/${task}.sh
source ./scripts/e3/cell.sh

date

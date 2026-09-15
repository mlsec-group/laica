#!/bin/bash
#SBATCH --job-name=laica-e1-n_param
#SBATCH --partition=gpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=data/log/laica-e1-n_param/array_%A-%a.out
#SBATCH --array=0-199
pwd; hostname; date

source ./scripts/e1/config.sh

task=n_param

source ./scripts/params/${task}.sh
source ./scripts/e1/cell.sh

date

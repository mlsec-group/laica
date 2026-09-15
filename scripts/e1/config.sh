###
### General
###

slurm=0
use_gpu=0

# When using slurm, make sure that the 'cpus-per-task' are set accordingly in the sbatch scripts
nprocs=16

###
### Paths
###

data_dir=/home/$USER/laica/data
tmp_dir=/tmp/$USER/laica/tmp
cache_dir=/tmp/$USER/laica/cache

###
### Train
###

# Only used when training without SLURM
cuda_visible_devices=0

# Resolve the tuned parameters from the search output instead of the cached tables
use_best_params=0

# Capping the validation and test splits. Leave empty to evaluate on all of them.
max_eval_samples=

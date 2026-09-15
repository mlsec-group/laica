source ./scripts/e1/config.sh

###
### Transformation search
###

search_seed=0
search_fraction=64
search_learning_rate=0.003
search_num_augmentations=3
trials_per_worker=28

###
### Grids
###

grid_learning_rates=(0.002 0.003 0.004)
grid_augmentation_runs=(1 3 5)

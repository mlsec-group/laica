###
### Tuned parameters for the n_param task, from the search over the three shared transformations
###

fractions=(1 2 4 8 16 32 64 128 256 512)

declare -A learning_rates=(
    [1]=0.002 [2]=0.003 [4]=0.004 [8]=0.004 [16]=0.004
    [32]=0.003 [64]=0.003 [128]=0.004 [256]=0.003 [512]=0.004
)

declare -A augmentation_runs=(
    [1]=3 [2]=5 [4]=3 [8]=5 [16]=1
    [32]=3 [64]=1 [128]=5 [256]=5 [512]=5
)

augmentation_params=(
    --p-reorder 0.7630989695621772
    --p-substitute 0.046469667078255554
    --p-insert-junk-code 0.025488488003974463
)

task_params=(
)

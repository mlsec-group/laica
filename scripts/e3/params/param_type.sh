###
### Tuned parameters for the param_type task, from the search over the three shared transformations
###

fractions=(1 2 4 8 16 32 64 128 256 512)

declare -A learning_rates=(
    [1]=0.003 [2]=0.004 [4]=0.003 [8]=0.004 [16]=0.003
    [32]=0.002 [64]=0.003 [128]=0.002 [256]=0.003 [512]=0.002
)

declare -A augmentation_runs=(
    [1]=3 [2]=5 [4]=1 [8]=5 [16]=5
    [32]=3 [64]=3 [128]=5 [256]=3 [512]=5
)

augmentation_params=(
    --p-reorder 0.4260606480904533
    --p-substitute 0.22377305266219466
    --p-insert-junk-code 0.5604330840734472
)

task_params=(
)

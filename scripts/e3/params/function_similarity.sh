###
### Tuned parameters for the function_similarity task, from the search over the three shared transformations
###

fractions=(1 2 4 8 16 32 64 128 256 512)

declare -A learning_rates=(
    [1]=0.002 [2]=0.002 [4]=0.004 [8]=0.003 [16]=0.003
    [32]=0.004 [64]=0.002 [128]=0.002 [256]=0.002 [512]=0.003
)

declare -A augmentation_runs=(
    [1]=1 [2]=5 [4]=5 [8]=5 [16]=5
    [32]=1 [64]=3 [128]=5 [256]=5 [512]=5
)

augmentation_params=(
    --p-reorder 0.02625469367766614
    --p-substitute 0.4394028629489356
    --p-insert-junk-code 0.1354263600179568
)

task_params=(
    --batch-size 2048
    --dimension 128
    --n-epochs-asm2vec-model 10

)

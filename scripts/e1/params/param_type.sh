###
### Tuned parameters for the param_type task
###

fractions=(1 2 4 8 16 32 64 128 256 512)
variants=(baseline augmented)

declare -A learning_rate_baseline=(
    [1]=0.004 [2]=0.002 [4]=0.002 [8]=0.004 [16]=0.004
    [32]=0.004 [64]=0.003 [128]=0.002 [256]=0.002 [512]=0.002
)

declare -A learning_rate_augmented=(
    [1]=0.002 [2]=0.003 [4]=0.004 [8]=0.003 [16]=0.002
    [32]=0.003 [64]=0.003 [128]=0.002 [256]=0.004 [512]=0.004
)

declare -A num_augmentations=(
    [1]=5 [2]=5 [4]=5 [8]=3 [16]=5
    [32]=3 [64]=5 [128]=5 [256]=3 [512]=3
)

augmentation_params=(
    --p-reorder 0.18814966515844211
    --p-substitute 0.7956553828708534
    --p-insert-junk-code 0.965091104263938
    --p-swap 0.7398551474039528
    --p-noise 0.7672220508334419
    --instruction-count-cropping 0
    --instruction-count-opaque-instruction-insertion 6
    --instruction-count-random-erasing 0
)

task_params=(

)

search_task_params=(
)

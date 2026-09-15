###
### Tuned parameters for the n_param task
###

fractions=(1 2 4 8 16 32 64 128 256 512)
variants=(baseline augmented)

declare -A learning_rate_baseline=(
    [1]=0.004 [2]=0.004 [4]=0.003 [8]=0.003 [16]=0.003
    [32]=0.004 [64]=0.004 [128]=0.004 [256]=0.003 [512]=0.004
)

declare -A learning_rate_augmented=(
    [1]=0.004 [2]=0.003 [4]=0.003 [8]=0.004 [16]=0.004
    [32]=0.003 [64]=0.002 [128]=0.003 [256]=0.004 [512]=0.004
)

declare -A num_augmentations=(
    [1]=1 [2]=3 [4]=1 [8]=1 [16]=3
    [32]=5 [64]=3 [128]=5 [256]=5 [512]=5
)

augmentation_params=(
    --p-reorder 0.7397278885942797
    --p-substitute 0.15651328876272497
    --p-insert-junk-code 0.046590317212061666
    --p-swap 0.7242459436776181
    --p-noise 0.8695562158422618
    --instruction-count-cropping 2
    --instruction-count-opaque-instruction-insertion 0
    --instruction-count-random-erasing 1
)

task_params=(

)

search_task_params=(
)

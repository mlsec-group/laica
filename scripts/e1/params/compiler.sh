###
### Tuned parameters for the compiler task
###

fractions=(1 2 4 8 16 32 64 128 256 512)
variants=(baseline augmented)

declare -A learning_rate_baseline=(
    [1]=0.002 [2]=0.003 [4]=0.004 [8]=0.003 [16]=0.003
    [32]=0.003 [64]=0.004 [128]=0.004 [256]=0.002 [512]=0.003
)

declare -A learning_rate_augmented=(
    [1]=0.002 [2]=0.002 [4]=0.002 [8]=0.002 [16]=0.003
    [32]=0.004 [64]=0.004 [128]=0.004 [256]=0.003 [512]=0.003
)

declare -A num_augmentations=(
    [1]=1 [2]=5 [4]=1 [8]=3 [16]=5
    [32]=3 [64]=1 [128]=5 [256]=5 [512]=5
)

augmentation_params=(
    --p-reorder 0.2307265235901876
    --p-substitute 0.5459814298981127
    --p-insert-junk-code 0.05883921528331453
    --p-swap 0.26063429052185555
    --p-noise 0.39841279762762627
    --instruction-count-cropping 4
    --instruction-count-opaque-instruction-insertion 9
    --instruction-count-random-erasing 3
)

task_params=(
    --batch-size 2048
)

search_task_params=(
)

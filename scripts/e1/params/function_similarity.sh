###
### Tuned parameters for the function_similarity task
###

fractions=(1 2 4 8 16 32 64 128 256 512)
variants=(baseline augmented)

declare -A learning_rate_baseline=(
    [1]=0.002 [2]=0.004 [4]=0.002 [8]=0.003 [16]=0.003
    [32]=0.002 [64]=0.002 [128]=0.002 [256]=0.002 [512]=0.003
)

declare -A learning_rate_augmented=(
    [1]=0.003 [2]=0.003 [4]=0.002 [8]=0.003 [16]=0.004
    [32]=0.003 [64]=0.002 [128]=0.002 [256]=0.004 [512]=0.003
)

declare -A num_augmentations=(
    [1]=3 [2]=5 [4]=1 [8]=3 [16]=3
    [32]=5 [64]=3 [128]=5 [256]=1 [512]=5
)

augmentation_params=(
    --p-reorder 0.06488448758510947
    --p-substitute 0.5472328819023279
    --p-insert-junk-code 0.19611502800117867
    --p-swap 0.7441045532970918
    --p-noise 0.018970348764023714
    --instruction-count-cropping 0
    --instruction-count-opaque-instruction-insertion 4
    --instruction-count-random-erasing 3
)

task_params=(
    --batch-size 2048
    --dimension 128
    --n-epochs-asm2vec-model 10
)

search_task_params=(
    --batch-size 2048
    --dimension 128
)

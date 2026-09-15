# laica

Code for the experiments in the paper: E1 (augmentation across dataset sizes), E2 (importance of each transformation) and E3 (relaxed vs. strict semantics).

## Setup

Everything runs inside an Apptainer container:

```bash
apptainer build container.sif container.def
```

Paths and SLURM settings are in `scripts/e1/config.sh`. The datasets are downloaded on first use from `https://huggingface.co/collections/mlsec-group/laica-datasets`.

Each experiment script is an array job with one run per index. To run a single one without SLURM, set `slurm=0` and pick the index yourself:

```bash
SLURM_ARRAY_TASK_ID=0 bash scripts/e1/n_param.sh
```

## E1

```bash
sbatch scripts/e1/n_param.sh    # also param_type, function_similarity, compiler
```

Trains and tests every combination of dataset fraction (1 to 1/512), baseline or augmented, and 10 seeds. The learning rates, number of augmentations and transformation parameters come from `scripts/e1/params/<task>.sh`. Results land in `data/e1/<task>/<variant>/<fraction>/<seed>/`.

## Hyperparameter search

```bash
sbatch scripts/e1/search/transformations.sh <task> # transformation parameters, Optuna
sbatch scripts/e1/search/runs.sh <task>            # learning rate x number of augmentations
sbatch scripts/e1/search/baseline.sh <task>        # learning rate without augmentation
```

Run the transformation search before `runs.sh`. With `use_best_params=1`, E1 and `runs.sh` use the search results instead of `scripts/e1/params`.

## E2

```bash
bash scripts/e2/e2.sh
```

Computes the fANOVA importance of each transformation from the transformation studies and writes `data/results/importances.csv`.

## E3

```bash
sbatch scripts/e3/n_param.sh    # also param_type, function_similarity
```

Compares the relaxed transformations, applied on the fly, with the strict LLVM-based ones, downloaded from `mlsec-group/laica-sp`. Both use the parameters in `scripts/e3/params/<task>.sh`, from the search over the three shared transformations. Results land in `data/e3/`.

## Evaluation

```bash
bash scripts/e1/evaluate.sh
```

Collects the E1 results into `data/results/results.csv` (one row per run) and `data/results/summary.csv` (mean, std and confidence interval per fraction).

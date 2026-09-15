from pathlib import Path

import click
from datasets import DatasetDict

from best_params import best_training_params, best_transformations
from augmentation.basic_block_reordering import BasicBlockReordering
from augmentation.cropping import Cropping
from augmentation.instruction_substitution import InstructionSubstitution
from augmentation.instruction_swapping import InstructionSwapping
from augmentation.junk_code_insertion import JunkCodeInsertion
from augmentation.opaque_instruction_insertion import OpaqueInstructionInsertion
from augmentation.random_erasing import RandomErasing
from augmentation.random_noise import RandomNoise
from cell import cell_options, run_cell
from config import TASKS
from dataset.dataset import augment_train, label_dataset, load_task_dataset, take_fraction
from util import log


def create_transformations(seed, p_reorder, p_insert_junk_code, p_substitute, p_swap, p_noise,
                           instruction_count_cropping, instruction_count_opaque_instruction_insertion,
                           instruction_count_random_erasing):
    return [
        BasicBlockReordering(seed=seed, p_reorder=p_reorder, keep_first_block=True),
        JunkCodeInsertion(seed=seed, p_insert_junk_code=p_insert_junk_code),
        InstructionSubstitution(seed=seed, p_substitute=p_substitute),
        InstructionSwapping(seed=seed, p_swap=p_swap),
        OpaqueInstructionInsertion(seed=seed, instruction_count=instruction_count_opaque_instruction_insertion),
        RandomNoise(seed=seed, p_noise=p_noise),
        RandomErasing(seed=seed, instruction_count=instruction_count_random_erasing),
        Cropping(seed=seed, instruction_count=instruction_count_cropping),
    ]


@click.command()
@click.option("--task", "-t", required=True, type=click.Choice(TASKS))
@click.option("--fraction", "-f", default=1, type=int, show_default=True, help="Train on 1/fraction of the training split.")
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--data-path", "-d", required=True, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Output directory.")
@click.option("--cache-dir", "-c", default="/tmp/cache", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Dataset cache directory.")
@click.option("--nprocs", "-p", default=1, type=int, show_default=True, help="Number of parallel processes for dataset processing.")
@click.option("--learning-rate", default=0.003, type=float, show_default=True)
@click.option("--num-augmentations", default=0, type=int, show_default=True, help="Number of augmented copies of the training split. 0 trains the baseline.")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Write the results here instead of the derived e1 directory.")
@click.option("--transformations-from-search", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Resolve the augmentation parameters from the transformation study in this search directory.")
@click.option("--training-params-from-search", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Resolve the learning rate and number of augmentations from the grid results in this search directory.")
@click.option("--p-reorder", default=0.0, type=float, show_default=True)
@click.option("--p-insert-junk-code", default=0.0, type=float, show_default=True)
@click.option("--p-substitute", default=0.0, type=float, show_default=True)
@click.option("--p-swap", default=0.0, type=float, show_default=True)
@click.option("--p-noise", default=0.0, type=float, show_default=True)
@click.option("--instruction-count-cropping", default=0, type=int, show_default=True)
@click.option("--instruction-count-opaque-instruction-insertion", default=0, type=int, show_default=True)
@click.option("--instruction-count-random-erasing", default=0, type=int, show_default=True)
@cell_options
def run(task, fraction, seed, data_path, cache_dir, nprocs, learning_rate, num_augmentations, output_dir,
        transformations_from_search, training_params_from_search, p_reorder, p_insert_junk_code, p_substitute,
        p_swap, p_noise, instruction_count_cropping, instruction_count_opaque_instruction_insertion,
        instruction_count_random_erasing, **cell_params):
    variant = "augmented" if num_augmentations > 0 else "baseline"

    if output_dir is None:
        output_dir = data_path / "e1" / task / variant / str(fraction) / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if transformations_from_search is not None:
        resolved = best_transformations(transformations_from_search, task)
        log(f"[+] Transformations from search: {resolved}")
        p_reorder = resolved["p_reorder"]
        p_insert_junk_code = resolved["p_insert_junk_code"]
        p_substitute = resolved["p_substitute"]
        p_swap = resolved["p_swap"]
        p_noise = resolved["p_noise"]
        instruction_count_cropping = resolved["instruction_count_cropping"]
        instruction_count_opaque_instruction_insertion = resolved["instruction_count_opaque_instruction_insertion"]
        instruction_count_random_erasing = resolved["instruction_count_random_erasing"]

    if training_params_from_search is not None:
        resolved = best_training_params(training_params_from_search, task, variant, fraction)
        log(f"[+] Training parameters from search: {resolved} (cached: learning rate {learning_rate}, {num_augmentations} augmentations)")
        learning_rate = resolved["learning_rate"]
        num_augmentations = resolved["num_augmentations"]

    log(f"[+] Running {task} {variant} 1/{fraction} seed {seed} into {output_dir}")

    dataset = load_task_dataset(task, cache_dir)
    dataset = label_dataset(dataset, task, nprocs)

    train_set = take_fraction(dataset["train"], fraction, seed, nprocs)

    if num_augmentations > 0:
        transformations = create_transformations(
            seed=seed,
            p_reorder=p_reorder,
            p_insert_junk_code=p_insert_junk_code,
            p_substitute=p_substitute,
            p_swap=p_swap,
            p_noise=p_noise,
            instruction_count_cropping=instruction_count_cropping,
            instruction_count_opaque_instruction_insertion=instruction_count_opaque_instruction_insertion,
            instruction_count_random_erasing=instruction_count_random_erasing,
        )
        log(f"[+] Augmenting the training split {num_augmentations} times")
        train_set = augment_train(train_set, transformations, num_augmentations, nprocs)

    dataset = DatasetDict({
        "train": train_set,
        "validation": dataset["validation"],
        "test": dataset["test"],
    })

    description = {
        "task": task,
        "variant": variant,
        "fraction": fraction,
        "seed": seed,
        "learning_rate": learning_rate,
        "num_augmentations": num_augmentations,
    }

    run_cell(dataset, output_dir, description, nprocs, **cell_params)


if __name__ == "__main__":
    run()

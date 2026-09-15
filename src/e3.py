from pathlib import Path

import click
from datasets import DatasetDict

from augmentation.basic_block_reordering import BasicBlockReordering
from augmentation.instruction_substitution import InstructionSubstitution
from augmentation.junk_code_insertion import JunkCodeInsertion
from cell import cell_options, run_cell
from dataset.dataset import augment_train, label_dataset, load_strict_dataset, load_task_dataset, take_fraction
from util import log

TASKS = ["function_similarity", "n_param", "param_type"]

VARIANTS = ["relaxed", "strict"]


def create_transformations(seed, p_reorder, p_insert_junk_code, p_substitute):
    return [
        BasicBlockReordering(seed=seed, p_reorder=p_reorder, keep_first_block=True),
        JunkCodeInsertion(seed=seed, p_insert_junk_code=p_insert_junk_code),
        InstructionSubstitution(seed=seed, p_substitute=p_substitute),
    ]


@click.command()
@click.option("--task", "-t", required=True, type=click.Choice(TASKS))
@click.option("--variant", required=True, type=click.Choice(VARIANTS), help="relaxed augments on the fly, strict reads the LLVM-augmented functions.")
@click.option("--fraction", "-f", default=1, type=int, show_default=True, help="Train on 1/fraction of the training split.")
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--data-path", "-d", required=True, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Output directory.")
@click.option("--cache-dir", "-c", default="/tmp/cache", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Dataset cache directory.")
@click.option("--nprocs", "-p", default=1, type=int, show_default=True, help="Number of parallel processes for dataset processing.")
@click.option("--learning-rate", default=0.003, type=float, show_default=True)
@click.option("--num-augmentations", required=True, type=click.IntRange(min=1), help="Number of augmented copies. The strict variant selects the set built with this many copies.")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Write the results here instead of the derived e3 directory.")
@click.option("--p-reorder", default=0.0, type=float, show_default=True)
@click.option("--p-insert-junk-code", default=0.0, type=float, show_default=True)
@click.option("--p-substitute", default=0.0, type=float, show_default=True)
@cell_options
def run(task, variant, fraction, seed, data_path, cache_dir, nprocs, learning_rate, num_augmentations,
        output_dir, p_reorder, p_insert_junk_code, p_substitute, **cell_params):

    if output_dir is None:
        output_dir = data_path / "e3" / task / variant / str(fraction) / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[+] Running {task} {variant} 1/{fraction} seed {seed} into {output_dir}")

    if variant == "strict":
        dataset = load_strict_dataset(task, seed, num_augmentations, fraction, cache_dir, nprocs)
    else:
        dataset = load_task_dataset(task, cache_dir)
        dataset = label_dataset(dataset, task, nprocs)

        train_set = take_fraction(dataset["train"], fraction, seed, nprocs)

        transformations = create_transformations(
            seed=seed,
            p_reorder=p_reorder,
            p_insert_junk_code=p_insert_junk_code,
            p_substitute=p_substitute,
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

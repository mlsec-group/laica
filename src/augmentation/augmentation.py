import functools as ft
import random

from datasets import concatenate_datasets

from util import md5_hash


def augment_function(function, augmentations):
    augmented_function = function
    for augmentation in augmentations:
        augmented = augmentation.augment_function(augmented_function)
        if augmented is not None:
            augmented_function = augmented

    augmented_function = dict(augmented_function)
    augmented_function["augmented"] = md5_hash(augmented_function["bytes"]) != md5_hash(function["bytes"])

    return augmented_function


def augment_dataset(dataset, augmentations, n_runs, nprocs):
    augmented_sets = [dataset]

    for run in range(n_runs):
        for augmentation in augmentations:
            augmentation.rng = random.Random(augmentation.seed + run)

        augmented_set = dataset.map(
            ft.partial(augment_function, augmentations=augmentations),
            num_proc=nprocs,
            desc=f"Augmenting run {run + 1}/{n_runs}",
        )
        augmented_set = augmented_set.filter(lambda sample: sample["augmented"])
        augmented_sets.append(augmented_set.remove_columns("augmented"))

    return concatenate_datasets(augmented_sets)

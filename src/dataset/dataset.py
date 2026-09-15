import math
import random
import re
from base64 import b64decode, b64encode
from collections import defaultdict

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

from augmentation.augmentation import augment_dataset
from config import (
    COMPILER_LABELS,
    DATASETS,
    E3_DATASET,
    MAX_NUM_PARAMS,
    PAIRS_NEGATIVE_RATIO,
    PAIRS_OPT_LEVELS,
    PARAM_TYPE_LABELS,
)
from instruction_formatter import format_instructions
from util import log

PAIR_COLUMNS = ["id", "opt_level", "base64_encoded_ops", "low_pc"]

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True


def load_task_dataset(task, cache_dir):
    return load_dataset(DATASETS[task], cache_dir=str(cache_dir))


def param_type_label(parameter_list):
    if len(parameter_list) == 0:
        return -1

    type_name = parameter_list[0]
    if re.match("^DW_TAG_pointer_type.*", type_name):
        return PARAM_TYPE_LABELS["pointer"]
    if re.match("^DW_TAG_structure_type.*", type_name):
        return PARAM_TYPE_LABELS["struct"]
    if re.match("^DW_TAG_union_type.*", type_name):
        return PARAM_TYPE_LABELS["union"]
    if re.match("^DW_TAG_enumeration_type.*", type_name):
        return PARAM_TYPE_LABELS["enum"]
    if re.match("^DW_TAG.*", type_name):
        return -1
    if re.match(".*float.*|.*double.*", type_name):
        return PARAM_TYPE_LABELS["float"]
    if re.match(".*char.*", type_name):
        return PARAM_TYPE_LABELS["char"]
    if re.match(".*int.*|.*short.*|.*long.*|.*_Bool.*", type_name):
        return PARAM_TYPE_LABELS["int"]
    return -1


def label_dataset(dataset, task, nprocs):
    if task == "n_param":
        dataset = dataset.map(lambda sample: {"label": len(sample["parameter_list"])}, num_proc=nprocs)
        return dataset.filter(lambda sample: sample["label"] < MAX_NUM_PARAMS, num_proc=nprocs)

    if task == "param_type":
        dataset = dataset.map(lambda sample: {"label": param_type_label(sample["parameter_list"])}, num_proc=nprocs)
        return dataset.filter(lambda sample: sample["label"] >= 0, num_proc=nprocs)

    if task == "compiler":
        return dataset.map(lambda sample: {"label": COMPILER_LABELS[sample["compiler"]]}, num_proc=nprocs)

    return dataset


def num_classes(task):
    if task == "n_param":
        return MAX_NUM_PARAMS
    if task == "param_type":
        return len(PARAM_TYPE_LABELS)
    if task == "compiler":
        return len(COMPILER_LABELS)
    raise ValueError(f"No classes for task: {task}")


def take_fraction(dataset, fraction, seed, nprocs):
    if fraction <= 1:
        return dataset

    ids = sorted(set(dataset["id"]))
    random.Random(seed).shuffle(ids)
    kept_ids = set(ids[: math.ceil(len(ids) / fraction)])

    log(f"[+] Keeping {len(kept_ids)} of {len(ids)} function ids (1/{fraction})")

    return dataset.filter(lambda sample: sample["id"] in kept_ids, num_proc=nprocs)


def augment_train(dataset, transformations, num_augmentations, nprocs):
    columns = dataset.column_names

    dataset = dataset.map(
        lambda sample: {"bytes": b64decode(sample["base64_encoded_ops"])},
        num_proc=nprocs,
        desc="Decoding function bytes",
    )
    dataset = augment_dataset(dataset, transformations, num_augmentations, nprocs)

    return dataset.select_columns(columns)


def decode_ops(batch):
    return {
        "base64_encoded_ops": [b64encode(ops).decode("utf-8") for ops in batch["ops"]],
        "data": [format_instructions(list(md.disasm(ops, low_pc))) for ops, low_pc in zip(batch["ops"], batch["low_pc"])],
    }


def load_strict_dataset(task, seed, num_augmentations, fraction, cache_dir, nprocs):
    dataset = load_dataset(
        E3_DATASET,
        data_files={
            "train": f"originals/{task}/{seed}.parquet",
            "validation": f"validation/{task}/data.parquet",
            "test": f"test/{task}/data.parquet",
        },
        cache_dir=str(cache_dir),
    )
    dataset = label_dataset(dataset, task, nprocs)

    train_set = take_fraction(dataset["train"], fraction, seed, nprocs)
    kept_ids = set(train_set["id"])

    augmented = load_dataset(
        E3_DATASET,
        data_files=f"augmented/{task}/{seed}.parquet",
        split="train",
        cache_dir=str(cache_dir),
    )
    augmented = augmented.filter(lambda sample: sample["n_aug"] == num_augmentations, num_proc=nprocs)

    if len(augmented) == 0:
        raise ValueError(f"No {task} functions augmented with {num_augmentations} copies for seed {seed} in {E3_DATASET}")

    selected = augmented.filter(lambda sample: sample["id"] in kept_ids, num_proc=nprocs)
    log(f"[+] Keeping {len(selected)} of {len(augmented)} functions augmented with {num_augmentations} copies")

    if "label" in train_set.column_names:
        labels = dict(zip(train_set["id"], train_set["label"]))
        selected = selected.map(lambda sample: {"label": labels[sample["id"]]}, num_proc=nprocs)

    columns = [column for column in train_set.column_names if column in selected.column_names]

    splits = {
        "train": concatenate_datasets([train_set.select_columns(columns), selected.select_columns(columns)]),
        "validation": dataset["validation"].select_columns(columns),
        "test": dataset["test"].select_columns(columns),
    }

    return DatasetDict({
        split: part.map(decode_ops, batched=True, num_proc=nprocs, remove_columns=["ops"], desc=f"Decoding the {split} functions")
        for split, part in splits.items()
    })


def create_pairs(dataset, seed, opt_levels=None, negative_ratio=None):
    opt_levels = PAIRS_OPT_LEVELS if opt_levels is None else opt_levels
    dataset = dataset.select_columns(PAIR_COLUMNS)

    opt_0_functions = dataset.filter(lambda x: x["opt_level"] == "O0")
    opt0_ids = set(opt_0_functions["id"])

    opt_n_functions = dataset.filter(lambda x: x["opt_level"] in opt_levels and x["id"] in opt0_ids)

    opt_0_functions = opt_0_functions.sort("id")
    opt_n_functions = opt_n_functions.sort("id")

    def group_by_id(dataset):
        groups = defaultdict(list)
        for row in dataset:
            groups[row["id"]].append({
                "base64_encoded_ops": row["base64_encoded_ops"],
                "low_pc": row["low_pc"],
            })
        return groups

    opt0_by_id = group_by_id(opt_0_functions)
    optn_by_id = group_by_id(opt_n_functions)

    common_ids = sorted(opt0_by_id.keys() & optn_by_id.keys())
    all_ids = sorted(optn_by_id.keys())

    rng = random.Random(seed)
    pairs = []
    positive_count = 0
    negative_count = 0

    def get_negative_sample(current_id):
        neg_id = rng.choice(all_ids)
        while neg_id == current_id:
            neg_id = rng.choice(all_ids)
        return rng.choice(optn_by_id[neg_id])

    for fid in tqdm(common_ids, desc="Creating similarity pairs"):
        for f0 in opt0_by_id[fid]:
            f1 = rng.choice(optn_by_id[fid])
            pairs.append({
                "function_0": f0,
                "function_1": f1,
                "label": 1,
            })
            positive_count += 1

            pairs.append({
                "function_0": f0,
                "function_1": get_negative_sample(fid),
                "label": 0,
            })
            negative_count += 1

    if negative_ratio is not None:
        while negative_count < positive_count * negative_ratio:
            fid = rng.choice(common_ids)
            f0 = rng.choice(opt0_by_id[fid])
            pairs.append({
                "function_0": f0,
                "function_1": get_negative_sample(fid),
                "label": 0,
            })
            negative_count += 1

    log(f"[+] Created {positive_count} positive and {negative_count} negative pairs")

    return Dataset.from_list(pairs)


def get_function_similarity_dataset(dataset, seed):
    log("[+] Creating function similarity pairs")

    return DatasetDict({
        split: create_pairs(dataset[split], seed, negative_ratio=PAIRS_NEGATIVE_RATIO)
        for split in dataset.keys()
    })


def unique_functions(dataset):
    seen = set()
    functions = []

    for pair in tqdm(dataset, desc="Extracting unique functions"):
        for function in (pair["function_0"], pair["function_1"]):
            key = (function["low_pc"], function["base64_encoded_ops"])
            if key in seen:
                continue
            seen.add(key)
            functions.append(function)

    log(f"[+] Extracted {len(functions)} unique functions")

    return functions

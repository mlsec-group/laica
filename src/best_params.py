import json
import time
from pathlib import Path

import optuna
import sqlalchemy

from util import log

STUDY_ATTEMPTS = 10

TRANSFORMATION_PARAMS = [
    "p_reorder",
    "p_insert_junk_code",
    "p_substitute",
    "p_swap",
    "instruction_count_opaque_instruction_insertion",
    "p_noise",
    "instruction_count_random_erasing",
    "instruction_count_cropping",
]


def study_path(search_dir, task):
    return Path(search_dir) / "transformations" / task / "study.sqlite3"


def load_study(search_dir, task, create=False):
    path = study_path(search_dir, task)

    if not create and not path.exists():
        raise FileNotFoundError(f"No transformation study for {task} at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(STUDY_ATTEMPTS):
        try:
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{path}",
                engine_kwargs={"connect_args": {"timeout": 100}},
            )
            return optuna.create_study(
                direction="maximize",
                storage=storage,
                study_name=task,
                load_if_exists=True,
            )
        except sqlalchemy.exc.SQLAlchemyError as error:
            if attempt == STUDY_ATTEMPTS - 1:
                raise
            log(f"[*] Study not ready ({error.__class__.__name__}), retrying in {attempt + 1}s")
            time.sleep(attempt + 1)


def best_transformations(search_dir, task):
    study = load_study(search_dir, task)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(trials) == 0:
        raise ValueError(f"No completed trials in the transformation study for {task}")

    best = max(trials, key=lambda trial: trial.value)

    return {name: best.params[name] for name in TRANSFORMATION_PARAMS}


def grid_directory(search_dir, task, variant, fraction):
    kind = "runs" if variant == "augmented" else "baseline"
    return Path(search_dir) / kind / task / str(fraction)


def best_training_params(search_dir, task, variant, fraction):
    directory = grid_directory(search_dir, task, variant, fraction)

    if not directory.exists():
        raise FileNotFoundError(f"No {variant} grid results for {task} 1/{fraction} at {directory}")

    results = [json.loads(path.read_text()) for path in sorted(directory.glob("*/objective.json"))]

    if len(results) == 0:
        raise ValueError(f"No completed cells in {directory}")

    best = max(results, key=lambda result: result["objective"])

    return {
        "learning_rate": best["learning_rate"],
        "num_augmentations": best["num_augmentations"],
    }

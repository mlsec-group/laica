import json
import subprocess
import sys
from pathlib import Path

import click
import optuna

from best_params import load_study
from config import TASKS
from util import log


def suggest_transformation_params(trial):
    return {
        "p_reorder": trial.suggest_float("p_reorder", 0.01, 1),
        "p_insert_junk_code": trial.suggest_float("p_insert_junk_code", 0.01, 1),
        "p_substitute": trial.suggest_float("p_substitute", 0.01, 1),
        "p_swap": trial.suggest_float("p_swap", 0.01, 1),
        "instruction_count_opaque_instruction_insertion": trial.suggest_int("instruction_count_opaque_instruction_insertion", 0, 10),
        "p_noise": trial.suggest_float("p_noise", 0, 1),
        "instruction_count_random_erasing": trial.suggest_int("instruction_count_random_erasing", 0, 5),
        "instruction_count_cropping": trial.suggest_int("instruction_count_cropping", 0, 5),
    }


def cell_command(params, output_dir, task, data_path, cache_dir, nprocs, seed, fraction,
                 learning_rate, num_augmentations, e1_args):
    command = [
        sys.executable, str(Path(__file__).parent / "e1.py"),
        "--task", task,
        "--fraction", str(fraction),
        "--seed", str(seed),
        "--data-path", str(data_path),
        "--cache-dir", str(cache_dir),
        "--nprocs", str(nprocs),
        "--learning-rate", str(learning_rate),
        "--num-augmentations", str(num_augmentations),
        "--output-dir", str(output_dir),
        "--skip-test",
    ]

    for name, value in params.items():
        command += [f"--{name.replace('_', '-')}", str(value)]

    return command + list(e1_args)


def run_trial(trial, search_dir, task, **kwargs):
    params = suggest_transformation_params(trial)
    output_dir = Path(search_dir) / "transformations" / task / str(trial.number)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[+] Trial {trial.number}: {params}")

    subprocess.run(cell_command(params, output_dir, task, **kwargs), check=True)

    return json.loads((output_dir / "objective.json").read_text())["objective"]


@click.command()
@click.option("--task", "-t", required=True, type=click.Choice(TASKS))
@click.option("--data-path", "-d", required=True, type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--cache-dir", "-c", default="/tmp/cache", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--nprocs", "-p", default=1, type=int, show_default=True)
@click.option("--trials", default=1, type=int, show_default=True, help="Number of trials this worker evaluates.")
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--fraction", default=64, type=int, show_default=True)
@click.option("--learning-rate", default=0.003, type=float, show_default=True)
@click.option("--num-augmentations", default=3, type=int, show_default=True)
@click.argument("e1_args", nargs=-1, type=click.UNPROCESSED)
def run(task, data_path, trials, **kwargs):
    search_dir = data_path / "search"
    study = load_study(search_dir, task, create=True)

    for _ in range(trials):
        trial = study.ask()
        try:
            value = run_trial(trial, search_dir, task, data_path=data_path, **kwargs)
        except subprocess.CalledProcessError as error:
            log(f"[!] Trial {trial.number} failed: {error}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        study.tell(trial, value)
        log(f"[+] Trial {trial.number} scored {value}")

    log(f"[+] Study has {len(study.trials)} trials")


if __name__ == "__main__":
    run()

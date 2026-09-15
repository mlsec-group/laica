from pathlib import Path

import click
import optuna
from optuna.importance import FanovaImportanceEvaluator

from best_params import TRANSFORMATION_PARAMS, load_study
from config import TASKS
from evaluate_e1 import write_csv
from util import log

SEED = 0

COLUMNS = ["task", "parameter", "importance_percent", "trials"]


@click.command()
@click.option("--data-path", "-d", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), help="Data directory holding the search results.")
@click.option("--output-dir", "-o", default=None, type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Where to write the csv file.")
def run(data_path, output_dir):
    search_dir = data_path / "search"

    if output_dir is None:
        output_dir = data_path / "results"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = []

    for task in TASKS:
        study = load_study(search_dir, task)
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        trials = len(completed)

        if len({trial.value for trial in completed}) < 2:
            raise ValueError(f"The transformation study for {task} has {trials} completed trials without two different objectives, so there is no variance to attribute")

        importances = optuna.importance.get_param_importances(
            study,
            evaluator=FanovaImportanceEvaluator(seed=SEED),
            params=TRANSFORMATION_PARAMS,
        )

        log(f"[+] {task} ({trials} trials): " + ", ".join(f"{name} {importances[name] * 100:.0f}%" for name in TRANSFORMATION_PARAMS))

        for parameter in TRANSFORMATION_PARAMS:
            rows.append({
                "task": task,
                "parameter": parameter,
                "importance_percent": importances[parameter] * 100,
                "trials": trials,
            })

    write_csv(output_dir / "importances.csv", COLUMNS, rows)


if __name__ == "__main__":
    run()

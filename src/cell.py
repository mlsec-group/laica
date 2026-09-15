import click
from datasets import DatasetDict

from asm2vec.asm2vec_model import Asm2VecModel
from dataset.dataset import get_function_similarity_dataset, unique_functions
from experiment.compiler_experiment import CompilerExperiment
from experiment.function_similarity_experiment import FunctionSimilarityExperiment
from experiment.n_param_experiment import NParamExperiment
from experiment.param_type_experiment import ParamTypeExperiment
from util import log

EXPERIMENTS = {
    "n_param": NParamExperiment,
    "param_type": ParamTypeExperiment,
    "compiler": CompilerExperiment,
}

CELL_OPTIONS = [
    click.option("--batch-size", default=1024, type=int, show_default=True),
    click.option("--dimension", default=128, type=int, show_default=True),
    click.option("--seq-length", default=256, type=int, show_default=True),
    click.option("--max-features", default=256, type=int, show_default=True),
    click.option("--decay-rate", default=0.9, type=float, show_default=True),
    click.option("--max-epochs", default=1000, type=int, show_default=True),
    click.option("--min-delta", default=0.001, type=float, show_default=True),
    click.option("--patience", default=10, type=int, show_default=True),
    click.option("--max-eval-samples", default=None, type=int, help="Cap the validation and test splits."),
    click.option("--skip-test", is_flag=True, help="Train only, do not predict on the test split."),
    click.option("--vector-size-asm2vec-model", default=64, type=int, show_default=True),
    click.option("--window-size-asm2vec-model", default=5, type=int, show_default=True),
    click.option("--n-epochs-asm2vec-model", default=10, type=int, show_default=True),
    click.option("--max-length-asm2vec-model", default=250, type=int, show_default=True),
]


def cell_options(command):
    for option in reversed(CELL_OPTIONS):
        command = option(command)
    return command


def cap_split(dataset, max_samples, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def create_embedding_model(dataset, output_dir, seed, nprocs, max_length, window_size, vector_size, n_epochs):
    functions = unique_functions(dataset["train"])

    model = Asm2VecModel(output_dir / "asm2vec", seed, max_length, max(1, min(nprocs, len(functions))))

    log("[+] Training the Asm2Vec model")
    model.train(
        functions=functions,
        widow_size=window_size,
        vector_size=vector_size,
        n_epochs=n_epochs,
    )

    return model


def run_cell(dataset, output_dir, description, nprocs, batch_size, dimension, seq_length, max_features,
             decay_rate, max_epochs, min_delta, patience, max_eval_samples, skip_test,
             vector_size_asm2vec_model, window_size_asm2vec_model, n_epochs_asm2vec_model,
             max_length_asm2vec_model):
    task = description["task"]
    seed = description["seed"]

    if task == "function_similarity":
        dataset = get_function_similarity_dataset(dataset, seed)

    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": cap_split(dataset["validation"], max_eval_samples, seed),
        "test": cap_split(dataset["test"], max_eval_samples, seed),
    })

    run_params = {
        **description,
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
    }

    log(f"[+] Samples: {dict((split, len(dataset[split])) for split in dataset.keys())}")

    experiment_args = dict(
        dataset=dataset,
        output_dir=output_dir,
        seed=seed,
        dimension=dimension,
        seq_length=seq_length,
        max_features=max_features,
        initial_learning_rate=description["learning_rate"],
        decay_rate=decay_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        min_delta=min_delta,
        patience=patience,
    )

    if task == "function_similarity":
        embedding_model = create_embedding_model(
            dataset=dataset,
            output_dir=output_dir,
            seed=seed,
            nprocs=nprocs,
            max_length=max_length_asm2vec_model,
            window_size=window_size_asm2vec_model,
            vector_size=vector_size_asm2vec_model,
            n_epochs=n_epochs_asm2vec_model,
        )
        experiment = FunctionSimilarityExperiment(
            embedding_model=embedding_model,
            max_length=max_length_asm2vec_model,
            **experiment_args,
        )
    else:
        experiment = EXPERIMENTS[task](**experiment_args)

    model = experiment.train()

    experiment.write_objective(run_params)

    if not skip_test:
        experiment.test(model, run_params)

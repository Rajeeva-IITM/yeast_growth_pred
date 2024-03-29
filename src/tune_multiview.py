import logging
from os import makedirs
from pathlib import Path

import hydra
import optuna
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.pretty import pprint
from rich.traceback import install

from data import KFoldEncodeModule
from lightning_model import NetMultiViewLightning

install()
console = Console(record=True)

torch.set_float32_matmul_precision("high")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def objective(trial: optuna.Trial, conf: DictConfig) -> float:
    """This function is the objective function for the Optuna optimization. It takes two
    parameters:

    - trial: An instance of the optuna.Trial class, which is used to sample hyperparameters for the model.
    - conf: A configuration dictionary that contains various settings for the optimization.

    The function returns a float value, which represents the average score of the model across multiple folds.

    The function first sets the random seed and empties the CUDA cache. Then, it defines the input and output sizes for the model.

    Next, it builds the layers before the concatenation by sampling the number of layers and the hidden size for each layer from the trial. The hidden size is updated in each iteration by dividing it by 2.

    After that, it builds the layers after the concatenation in a similar way.

    The function then samples the learning parameters from the trial, including the learning rate, weight decay, maximum learning rate, dropout rate, and activation function.

    It prints a message indicating the start of the run.

    Next, it performs k-fold cross-validation. For each fold, it initializes the datamodule and the model with the sampled hyperparameters.

    It then instantiates the trainer object from the configuration and fits the model using the datamodule.

    After each fold, it appends the score of the model to the score_vector.

    Finally, it prints a message indicating the end of the run, and returns the mean of the score_vector as a float value.
    """
    seed_everything(42)
    torch.cuda.empty_cache()

    input_size = [6014, 64]
    output_size = 1

    # Building view1 and view2 layers
    num_layers_before = trial.suggest_int("num_layers_before", 2, 5)
    hidden_size_before = trial.suggest_categorical("hidden_size_before", [256, 512, 1024, 2048])

    hidden_layers_before = []
    for _ in range(num_layers_before):
        hidden_layers_before.append(hidden_size_before)
        hidden_size_before = hidden_size_before // 2

    # Building postconcatenation layers
    num_layers_after = trial.suggest_int("num_layers_after", 3, 5)
    hidden_size_after = trial.suggest_categorical("hidden_size_after", [256, 512, 1024, 2048])
    hidden_layers_after = []

    for _ in range(num_layers_after):
        hidden_layers_after.append(hidden_size_after)
        hidden_size_after = hidden_size_after // 2

    # Learning parameters
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    max_lr = trial.suggest_float("max_lr", 1e-3, 1e-1)
    dropout = trial.suggest_float("dropout", 0, 1)
    activation = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid", "celu", "gelu"]
    )

    console.print("Starting Run", justify="center")

    # k-Fold Cross Validation
    score_vector = []

    for k in range(5):
        console.log(f"Fold {k}", justify="center")

        datamodule = KFoldEncodeModule(
            conf.data.datamodule.path,
            k=k,
            split_seed=conf.data.datamodule.split_seed,
            num_splits=conf.data.datamodule.num_splits,
            num_workers=conf.data.datamodule.num_workers,
            batch_size=conf.data.datamodule.batch_size,
            test_size=conf.data.datamodule.test_size,
            stratify=conf.data.datamodule.stratify,
        )

        # model
        model = NetMultiViewLightning(
            layers_before_concat=hidden_layers_before,
            layers_after_concat=hidden_layers_after,
            input_size=input_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
            lr=lr,
            max_lr=max_lr,
            weight_decay=weight_decay,
            task="binary_classification",
            loss_function="bce",
        )

        trainer: Trainer = hydra.utils.instantiate(conf.trainer)
        # trainer: Trainer = hydra.utils.instantiate(conf.trainer, callbacks=[pbar])

        trainer.fit(model, datamodule)

        score_vector.append(trainer.callback_metrics[conf.optuna.objective].item())

    console.rule(title="[bold red]THE RUN ENDS[/bold red]", characters="-*", align="center")

    return torch.Tensor(score_vector).mean().item()


@hydra.main(version_base=None, config_path="../configs/", config_name="tuning.yaml")
def main(conf: DictConfig):
    """Runs the main function with the provided configuration.

    Args:
        conf (DictConfig): The configuration for the function.

    Returns:
        None
    """
    pprint(conf)

    groupname = conf.data.metadata.groupname
    savename = conf.data.metadata.savename

    # pbar_theme: RichProgressBarTheme = hydra.utils.instantiate(conf.callbacks.get("rich_progress_bar"))
    # pbar = RichProgressBar(theme=pbar_theme)

    file_path = Path(conf.paths.log_dir) / groupname

    console.print(f"[royal_blue1]Optuna Tuning - {groupname}[/royal_blue1]")

    if not file_path.exists():
        makedirs(file_path, exist_ok=True)
        with open(file_path / f"{savename}.log", "w") as f:
            console.print(f"File at [sandy_brown]{file_path}[/sandy_brown] created.")

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(file_path / savename))
    )
    sampler = hydra.utils.instantiate(conf.get("optuna.sampler"))

    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        study_name=groupname,
        direction=conf.optuna.direction,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, conf),
        n_trials=conf.optuna.n_trials,
        gc_after_trial=True,
        n_jobs=conf.optuna.n_jobs,
    )

    console.print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial

    console.print(
        "Best trial:",
    )

    console.print(f"Value: {trial.value}")

    console.print("Params: ")
    console.print(trial.params)


if __name__ == "__main__":
    main()

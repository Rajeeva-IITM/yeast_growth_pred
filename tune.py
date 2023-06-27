# Trying Optuna for tuning 

import optuna
import torch
import wandb
from pathlib import Path
from torch import nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from optuna.integration import PyTorchLightningPruningCallback

from rich.console import Console
from rich.traceback import install

from lightning_model import Netlightning
from data import EncodeModule


install()
console = Console(record=True)

torch.set_float32_matmul_precision("high")

# Define filename
filename = "../data/regression_data/bloom2013_regression.feather"
groupname = "Tuning - Bloom2013"
date = "mwah"

pbar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="#af005f",
                progress_bar_finished="green_yellow",
                batch_progress="cyan1",
                time='bright_red',
                metrics="gold3"
            )
        )

datamodule = EncodeModule(filename, batch_size=64, num_workers=4)
datamodule.setup()

def objective(trial: optuna.Trial) -> float:
    
    seed_everything(42)
    torch.cuda.empty_cache()
    
    wandb_logger = WandbLogger(
    project="Regression - Bloom",
    group=groupname,
    job_type="Tuning",
    save_dir="/storage/bt20d204/runs/regression_bloom/tuning_2013/"
    )
    
    input_size = 6078
    output_size = 1
        
    num_layers = trial.suggest_int("num_layers", 2, 8)
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024, 2048, 4096])
    
    hidden_layers = []
    for _ in range(num_layers):
        hidden_layers.append(hidden_size)
        hidden_size = hidden_size // 2
        
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    max_lr = trial.suggest_float("max_lr", 1e-3, 1e-1)
    dropout = trial.suggest_float("dropout", 0, 1)
    activation = trial.suggest_categorical(
        "activation", ["relu", "tanh", "sigmoid", "celu", "gelu"]
    )
    
    model = Netlightning(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        max_lr=max_lr,
        activation = activation,
        loss_function = "mse",
    )
    
    trainer = Trainer(
    logger=wandb_logger, accelerator='gpu', devices=1, max_epochs=50, 
    callbacks=[pbar, PyTorchLightningPruningCallback(trial, monitor="val_mse")],
    enable_checkpointing=False
    )
    
    trainer.fit(model, datamodule)
    wandb.finish()
    
    return trainer.callback_metrics["val_mse"].item()

if __name__ == '__main__':
    
    filepath = Path(f"/storage/bt20d204/runs/regression_bloom/tuning_2013/{date}.log")
    
    console.print(
        f"[royal_blue1]Optuna Tuning - {groupname}[/royal_blue1]",
        justify="center",
    )
    
    if not filepath.exists():
        with filepath.open("w") as f:
            console.print("File at [sandy_brown]{} [/sandy_brown] created.".format(filepath))
    
    
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(filepath))
    )
    pruner = optuna.pruners.HyperbandPruner()
    sampler = optuna.samplers.TPESampler()
    
    study = optuna.create_study(
        storage=storage,
        pruner=pruner,
        sampler=sampler,
        study_name=groupname,
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=250)
    
    console.print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trial

    console.print("Best trial:", )

    console.print("Value: {}".format(trial.value))

    console.print("Params: ")
    console.print(trial.params)
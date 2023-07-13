import torch 
import wandb
from lightning_model import Netlightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger

from data import KFoldEncodeModule

torch.set_float32_matmul_precision("high")

name = "bloom2013_regression"
data_filename = "../data/regression_data/bloom2013_regression.feather"

if __name__ == "__main__":
    
    
    results = []
    num_folds = 5

    wandb_savedir = f"../runs/regression_bloom/{name}"
    ckpt_savedir = f"../runs/regression_bloom/{name}"
    ckpt_savename = f"{name}-" + "{epoch}-{step}"
    
    pbar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="#af005f",
                progress_bar_finished="green_yellow",
                batch_progress="cyan1",
                time='bright_red',
                metrics="gold3"
            )
        )
    
    for k in range(num_folds):
    
        print(f"Fold {k}")
        wandb_logger = WandbLogger(
            project="Regression - Bloom",
            name=f"Fold {k}",
            save_dir=wandb_savedir,
            job_type="KFold",
            group=f"{name}",
            notes=f"Dataset: {name}",
        )
    
        datamodule = KFoldEncodeModule(
            data_filename,
            k=k,
            split_seed=42,
            num_splits=5,
            num_workers=10,
            batch_size=64,
            test_size=0.2
        )
        
        datamodule.setup()
        
        model = Netlightning(
            input_size=6078,
            output_size=1,
            hidden_layers= [1024,512,256,128],
            dropout=0.8992,
            lr=0.00091,
            weight_decay=0.00306,
            max_lr=0.01003,
            activation="gelu",
            loss_function="mse",
        )
        
        checkpoint = ModelCheckpoint(
            monitor="val_r2",
            filename=ckpt_savename,
            mode="max",
            dirpath=ckpt_savedir,
        )
        
        trainer = Trainer(
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
            callbacks=[checkpoint, pbar],
        )
        
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path="best")
        wandb.finish()
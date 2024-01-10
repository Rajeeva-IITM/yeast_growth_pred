import hydra
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger

from data import KFoldEncodeModule
from lightning_model import NetMultiViewLightning

torch.set_float32_matmul_precision("high")

# name = "bloom2015_reg"
# data_filename = "../data.datamodule/regression_data/bloom2015_regression.feather"


@hydra.main(version_base=None, config_path="../configs/", config_name="train.yaml")
def main(conf: DictConfig):
    num_folds = conf.data.datamodule.num_splits

    pbar = RichProgressBar(theme=hydra.utils.instantiate(conf.callbacks.rich_progress_bar))

    for k in range(num_folds):
        print(f"Fold {k}")

        ckpt_savename = f"-{k}-".join(conf.callbacks.model_checkpoint.filename.split("-"))
        print(ckpt_savename)

        wandb_logger = hydra.utils.instantiate(conf.logging.wandb, name=f"Fold {k}")

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

        datamodule.setup()

        model = NetMultiViewLightning(
            layers_before_concat=conf.model.layers_before_concat,
            layers_after_concat=conf.model.layers_after_concat,
            input_size=conf.model.input_size,
            output_size=conf.model.output_size,
            dropout=conf.model.dropout,
            activation=conf.model.activation,
            lr=conf.model.lr,
            max_lr=conf.model.max_lr,
            weight_decay=conf.model.weight_decay,
            task=conf.model.task,
            loss_function=conf.model.loss_function,
        )

        checkpoint = ModelCheckpoint(
            monitor=conf.callbacks.model_checkpoint.monitor,
            filename=ckpt_savename,
            mode=conf.callbacks.model_checkpoint.mode,
            dirpath=conf.callbacks.model_checkpoint.dirpath,
        )

        trainer = hydra.utils.instantiate(
            conf.trainer, logger=wandb_logger, callbacks=[pbar, checkpoint]
        )

        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path="best")
        wandb.finish()


if __name__ == "__main__":
    main()

import hydra

# from data import KFoldEncodeModule
import rootutils
import torch
import wandb
from FGR.load_FGR import get_fgr_model

# from pytorch_lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint  # , EarlyStopping
from omegaconf import DictConfig

# from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
# from pytorch_lightning.loggers.wandb import WandbLogger


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from lightning_model import Netlightning

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../configs/", config_name="train.yaml")
def main(conf: DictConfig):
    """The main function that serves as the entry point for the program.

    Args:
        conf (DictConfig): The configuration object containing various settings.

    Returns:
        None
    """
    num_folds = conf.data.datamodule.num_splits

    # pbar = RichProgressBar(theme=hydra.utils.instantiate(conf.callbacks.rich_progress_bar))

    for k in range(num_folds):
        print(f"Fold {k}")

        ckpt_savename = f"-{k}-".join(conf.callbacks.model_checkpoint.filename.split("-"))
        print(ckpt_savename)

        wandb_logger = hydra.utils.instantiate(
            conf.logging.wandb, name=f"Fold {k}"
        )  # Initialize the logger

        datamodule = hydra.utils.instantiate(
            conf.data.datamodule, k=k
        )  # Initialize the datamodule

        datamodule.setup()

        if conf.model.fgr:
            fgr_model = get_fgr_model(conf.model.ckpt_path)
            model = hydra.utils.instantiate(conf.model, fgr_model=fgr_model)

        else:
            model = hydra.utils.instantiate(conf.model)  # Initialize the model

        # model = hydra.utils.instantiate(conf.model)  # Initialize the model

        callbacks = [
            hydra.utils.instantiate(cb) for cb in conf.callbacks if "model_checkpoint" not in cb
        ]

        checkpoint = ModelCheckpoint(
            monitor=conf.callbacks.model_checkpoint.monitor,
            filename=ckpt_savename,
            mode=conf.callbacks.model_checkpoint.mode,
            dirpath=conf.callbacks.model_checkpoint.dirpath,
        )

        trainer = hydra.utils.instantiate(
            conf.trainer, logger=wandb_logger, callbacks=[*callbacks, checkpoint]
        )

        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path="best")
        wandb.finish()


if __name__ == "__main__":
    main()

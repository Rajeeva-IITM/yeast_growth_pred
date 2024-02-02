import finetuning_scheduler
import hydra

# from data import KFoldEncodeModule
import rootutils
import torch
import wandb

# from pytorch_lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint  # , EarlyStopping
from omegaconf import DictConfig

# from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
# from pytorch_lightning.loggers.wandb import WandbLogger


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import sys

from main_code.FGR.load_FGR import get_fgr_model

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
    sys.path.append(conf.paths.FGR_dir)

    num_folds = conf.data.datamodule.num_splits

    fgr = conf.model.fgr
    ckpt_path = conf.model.ckpt_path
    del conf.model.fgr
    del conf.model.ckpt_path
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

        callbacks = [
            hydra.utils.instantiate(conf.callbacks[cb])
            for cb in conf.callbacks
            if cb not in ["fine_tune_scheduler", "model_checkpoint"]
        ]

        if fgr:
            fgr_model = get_fgr_model(ckpt_path)

            net = hydra.utils.instantiate(conf.model.net, fgr_model=fgr_model)
            # print(net)
            model = hydra.utils.instantiate(conf.model, net=net)

            fts = finetuning_scheduler.FinetuningScheduler(**conf.callbacks.fine_tune_scheduler)
            callbacks.append(fts)

        else:
            model = hydra.utils.instantiate(conf.model)  # Initialize the model

        # print(model)

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

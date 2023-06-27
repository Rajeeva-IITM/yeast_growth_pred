from typing import Callable, List

import torch
# from lightning.pytorch.callbacks import Callback
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from network import Net
from torch import nn

from data import EncodeModule


class Netlightning(LightningModule):
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        max_lr: float,
        activation: str,
        loss_function: str,
    ):
        """Initialize the Lightning model for regression.

        Args:
            input_size (int): size of the input layer
            output_size (int): size of the output layer
            hidden_layers (int): sizes of the hidden layers
            dropout (float): dropout
            lr (float): learning rate
            weight_decay (float): weigt_decay
            max_lr (float): maximal learning rate
            activation (str): Activation Function
            loss_function (str): Loss Function. Must be one of ["mse", "mae", "huber"]
        """
        
        super(Netlightning, self).__init__()
        
        self.save_hyperparameters(ignore=['loss_function'])
        
        self.net = Net(
            hidden_layers = hidden_layers,
            input_size=input_size,
            output_size=output_size,
            dropout=dropout,
            activation = activation
        )
        
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        
        self.train_exp_var = torchmetrics.ExplainedVariance()
        self.val_exp_var = torchmetrics.ExplainedVariance()
        self.test_exp_var = torchmetrics.ExplainedVariance()
        
        match loss_function:
            case "mse":
                self.criterion = nn.MSELoss()
            case "mae":
                self.criterion = nn.L1Loss()
            case "huber":
                self.criterion = nn.HuberLoss()
            case _:
                self.criterion = nn.MSELoss()
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,  
        )
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.train_mse.update(y_pred, y)
        self.train_r2.update(y_pred, y)
        self.train_exp_var.update(y_pred, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_mse",
            self.train_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_r2",
            self.train_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_exp_var",
            self.train_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        
        self.val_mse.update(y_pred, y)
        self.val_r2.update(y_pred, y)
        self.val_exp_var.update(y_pred, y)
        
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        self.log(
            "val_mse",
            self.val_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        self.log(
            "val_r2",
            self.val_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        self.log(
            "val_exp_var",
            self.val_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return None
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        
        self.test_mse.update(y_pred, y)
        self.test_r2.update(y_pred, y)
        self.test_exp_var.update(y_pred, y)
        
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        self.log(
            "test_mse",
            self.test_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        self.log(
            "test_r2",
            self.test_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        self.log(
            "test_exp_var",
            self.test_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return None
    
if __name__ == "__main__":
    
    torch.set_float32_matmul_precision("high")
    
    model = Netlightning(
        input_size=6078,
        output_size=1,
        hidden_layers=[100, 100],
        dropout=0.5,
        lr=1e-3,
        weight_decay=1e-5,
        max_lr=1e-2,
        activation="relu",
        loss_function="mse",
    )
    
    datamodule = EncodeModule(
        path="../data/regression_data/bloom2013_regression.feather"
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=10
    )
    
    trainer.fit(model, datamodule)
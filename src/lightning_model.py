from typing import Any, Callable, List
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
# from lightning.pytorch.callbacks import Callback
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from network import Net, MultiViewNet
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
            hidden_layers List(int): sizes of the hidden layers
            dropout (float): dropout
            lr (float): learning rate
            weight_decay (float): weigt_decay
            max_lr (float): maximal learning rate
            activation (str): Activation Function
            loss_function (str): Loss Function. Must be one of ["mse", "mae", "huber"]
        """
        
        super(Netlightning, self).__init__()
        
        self.save_hyperparameters()
        
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
    
class NetMultiViewLightning(LightningModule):
    
    def __init__(
        self,
        layers_before_concat: List[int],
        layers_after_concat: List[int],
        input_size: List[int],
        output_size: int,
        dropout: float,
        activation: str,
        lr: float,
        max_lr: float,
        weight_decay: float,
        task: str = "regression",
        loss_function: str = "mse",
    ):
        """_summary_

        Args:
            layers_before_concat (List[int]): _description_
            layers_after_concat (List[int]): _description_
            input_size (List[int]): _description_
            output_size (int): _description_
            dropout (float): _description_
            activation (str): _description_
            lr (float): _description_
            max_lr (float): _description_
            weight_decay (float): _description_
            task (str, optional): Task of the model. Defaults to "regression".
            loss_function (str, optional): _description_. Defaults to "mse".
        """
        
        super(NetMultiViewLightning, self).__init__()
        
        self.save_hyperparameters()
        
        if task == "regression":
            # Map loss function names to their respective loss functions
            loss_functions = {
                "mse": nn.MSELoss(),
                "mae": nn.L1Loss(),
                "huber": nn.HuberLoss()
            }
            
            # Check if the loss function is valid
            if loss_function not in loss_functions:
                raise ValueError("Loss function must be one of ['mse', 'mae', 'huber']")
            
            # Set the criterion based on the loss function
            self.criterion = loss_functions[loss_function]
            
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
            
            self.train_r2 = torchmetrics.R2Score()
            self.val_r2 = torchmetrics.R2Score()
            self.test_r2 = torchmetrics.R2Score()
        
        elif task == "binary_classification":
            
            loss_functions = {
                "bce": nn.BCEWithLogitsLoss(),
                "cross_entropy": nn.CrossEntropyLoss()
            }
            
            self.train_acc = torchmetrics.Accuracy(task="binary")
            self.val_acc = torchmetrics.Accuracy(task="binary")
            self.test_acc = torchmetrics.Accuracy(task="binary")
            
            self.train_auc_roc = torchmetrics.AUROC(task="binary")
            self.val_auc_roc = torchmetrics.AUROC(task="binary")
            self.test_auc_roc = torchmetrics.AUROC(task="binary")
            
            self.train_f1 = torchmetrics.F1Score(task="binary")
            self.val_f1 = torchmetrics.F1Score(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            
            if loss_function not in loss_functions:
                raise ValueError("Loss function must be one of ['bce', 'cross_entropy']")
            
            self.criterion = loss_functions[loss_function]
        
        else:
            raise ValueError("Task must be one of ['regression', 'binary_classification']")
        
        self.task = task
        
        #Defining model
        self.net = MultiViewNet(
            layers_before_concat=layers_before_concat,
            layers_after_concat=layers_after_concat,
            input_size=input_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
        )
        
        # Learning Parameters
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
            
            if self.task == "regression":
                self.train_r2.update(y_pred, y)
                self.train_mae.update(y_pred, y)
                
                self.log("train_r2", self.train_r2, on_epoch=True, prog_bar=True)
                self.log("train_mae", self.train_mae, on_epoch=True, prog_bar=True)
                
            elif self.task == "binary_classification":
                self.train_acc.update(y_pred, y)
                self.train_auc_roc.update(y_pred, y)
                self.train_f1.update(y_pred, y)
                
                self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
                self.log("train_auc_roc", self.train_auc_roc, on_epoch=True, prog_bar=True)
                self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
            
            self.log("train_loss", loss, prog_bar=True)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        
        
        # for metric_name in filter(lambda x: 'val' in x, self.metric_names):
        #     metric = getattr(self, metric_name)
        #     metric.update(y_pred, y)
        #     self.log(
        #         metric_name,
        #         metric,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True
        #     )
        
        if self.task == "regression":
            self.val_r2.update(y_pred, y)
            self.val_mae.update(y)
            self.log("val_r2", self.val_r2, on_epoch=True)
            self.log("val_mae", self.val_mae, on_epoch=True)
        
        elif self.task == "binary_classification":
            self.val_acc.update(y_pred, y)
            self.val_auc_roc.update(y_pred, y)
            self.val_f1.update(y_pred, y)
            self.log("val_acc", self.val_acc, on_epoch=True)
            self.log("val_auc_roc", self.val_auc_roc, on_epoch=True)
            self.log("val_f1", self.val_f1, on_epoch=True)
        
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
            
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        
        if self.task == "regression":
            self.test_r2.update(y_pred, y)
            self.test_mae.update(y_pred, y)
            
            self.log("test_r2", self.test_r2, on_epoch=True, prog_bar=True)
            self.log("test_mae", self.test_mae, on_epoch=True, prog_bar=True)
        
        elif self.task == "binary_classification":
            self.test_acc.update(y_pred, y)
            self.test_auc_roc.update(y_pred, y)
            self.test_f1.update(y_pred, y)
        
            self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
            self.log("test_auc_roc", self.test_auc_roc, on_epoch=True, prog_bar=True)
            self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)
        
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        
if __name__ == "__main__":
    
    torch.set_float32_matmul_precision("high")
    
    # model = Netlightning(
    #     input_size=6078,
    #     output_size=1,
    #     hidden_layers=[100, 100],
    #     dropout=0.5,
    #     lr=1e-3,
    #     weight_decay=1e-5,
    #     max_lr=1e-2,
    #     activation="relu",
    #     loss_function="mse",
    # )
    
    # Small test
    
    model = NetMultiViewLightning(
        layers_before_concat=[100, 100],
        layers_after_concat=[100, 100],
        input_size=[6014, 64],
        output_size = 1,
        dropout=0.5,
        activation="relu",
        lr=1e-3,
        max_lr=1e-2,
        weight_decay=1e-5,
        task="binary_classification",
        loss_function="bce"
    )
    
    # datamodule = EncodeModule(
    #     path = "/storage/bt20d204/data/regression_data/bloom2013_regression.feather"
    # )
    
    datamodule = EncodeModule(
        path="/storage/bt20d204/data/bloom2013_clf_3_pubchem.feather",
        num_workers=1
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=100
    )
    
    trainer.fit(model, datamodule)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchmetrics import AUROC, Accuracy, F1Score, MeanSquaredError, R2Score


def set_theme() -> None:
    """Consistency is Key."""
    sns.set_theme(
        context="notebook",
        style="white",
        palette="colorblind",
        rc={
            "figure.figsize": (8, 6),
            "xtick.labelsize": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "ytick.labelsize": 12,
            "figure.dpi": 120,
        },
    )

    sns.set_palette(
        [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]
    )
    np.set_printoptions(precision=4)


def predict(model, X, clf=False, device="cuda:0"):
    y_pred = model(X.to(device))
    if clf:
        y_pred = torch.argmax(y_pred, dim=1)
    return y_pred

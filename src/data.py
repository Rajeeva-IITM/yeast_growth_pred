from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import torch
from pytorch_lightning import LightningDataModule
from rich.console import Console
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

console = Console()
ArrayLike = Union[List, tuple, np.ndarray]
DataFrameLike = Union[pd.DataFrame, pl.DataFrame]


class EncodeDataset(Dataset):
    """Dataset used by Pytorch.

    Args:
        X (np.array): Input Variable
        y (np.array): Target Variable
        labels (Iterable, optional): Labels for the dataset. Defaults to None.
    """

    def __init__(self, X, y, labels: Iterable = None) -> None:
        assert len(X) == len(y), "X, y must have the same length."
        # assert X.dtype == np.float32 and y.dtype == np.float32, "X, y must be of type np.float32."

        self.X = X
        self.y = y

        self.size = len(X)
        self.labels = labels

    def __len__(self):
        """Returns the length of the object.

        :return: The length of the object.
        :rtype: int
        """
        return self.size

    def __getitem__(self, index) -> Any:
        """Get the item at the given index from the X and y arrays.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: A tuple of two elements.
        """
        return self.X.iloc[index].astype(np.float32), np.asarray(self.y[index]).reshape(1)


class KFoldEncodeModule(LightningDataModule):
    """Lightning data module for K-Fold Cross Validation.

    Args:
        path (str): path to the dataset
        format (str, optional): dataset format. Defaults to None.
        k (int, optional): index of the k-fold. Defaults to 0.
        split_seed (int, optional): Random seed. Defaults to 42.
        num_splits (int, optional): Number of Splits. Defaults to 5.
        num_workers (int, optional): Number of workers. Defaults to 4.
        batch_size (int, optional): batch size. Defaults to 64.
        test_size (float, optional): test size. Defaults to 0.2.
        stratify (str, optional): stratify column. Defaults to None.
    """

    def __init__(
        self,
        path: str,
        format: str = None,
        k: int = 0,
        split_seed: int = 42,
        num_splits: int = 5,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
        stratify: "str" = None,
    ):
        super().__init__()
        self.path = path

        if format is None:
            self.format = path.split(".")[-1]
        else:
            self.format = format

        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size
        self.stratify = stratify

        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage=None):
        """Set up the data for the experiment.

        Args:
            stage (optional): The stage of the setup process. Defaults to None.

        Raises:
            NotImplementedError: If the file format is not supported.

        Returns:
            None
        """
        # print("Setting up data")

        match self.format:
            case "feather":
                data = pd.read_feather(self.path)
            case "parquet":
                data = pd.read_parquet(self.path)
            case "csv":
                data = pd.read_csv(self.path)
            case _:
                raise NotImplementedError(
                    "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
                )

        self.X = data.drop(
            ["Phenotype", "Condition", "Strain"], axis=1
        )  # TODO: Need to generalize
        self.y = data["Phenotype"].astype(np.float32)
        self.stratify_col = data[self.stratify] if self.stratify else None

        try:
            self.condition = data["Condition"]
        except:  # noqa: E722
            self.condition = None

        self.dataset = EncodeDataset(self.X.values, self.y.values)

        del self.X, self.y, data

        train_indices, test_indices = train_test_split(
            np.arange(len(self.dataset)),
            test_size=self.test_size,
            random_state=self.split_seed,
            stratify=self.stratify_col,
        )

        # Create test and training datasets
        self.test_dataset = Subset(self.dataset, test_indices)
        self.dataset_for_split = Subset(self.dataset, train_indices)

        if not self.train_dataset and not self.val_dataset:
            # Create KFold object
            kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)

            # Split data into training and validation subsets
            all_splits = list(kf.split(self.dataset_for_split))
            train_index, val_index = all_splits[self.k]

            # Create training and validation datasets
            self.train_dataset = Subset(self.dataset_for_split, train_index)
            self.val_dataset = Subset(self.dataset_for_split, val_index)

    def train_dataloader(self):
        """Returns a PyTorch DataLoader object for the training dataset.

        Parameters:
            None

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Generate a validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Generate a test dataloader.

        Returns:
            DataLoader: The DataLoader object created with the specified parameters.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class EncodeModule(LightningDataModule):
    """Lightning data module for K-Fold Cross Validation.

    Args:
        path (str): path to the dataset
        format (str, optional): dataset format. Defaults to "feather".
        k (int, optional): index of the k-fold. Defaults to 0.
        split_seed (int, optional): Random seed. Defaults to 42.
        num_splits (int, optional): Number of Splits. Defaults to 5.
        num_workers (int, optional): Number of workers. Defaults to 4.
        batch_size (int, optional): batch size. Defaults to 64.
        test_size (float, optional): test size. Defaults to 0.2.
    """

    def __init__(
        self,
        path: str,
        format: str = "feather",
        split_seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
    ):
        super().__init__()

        self.path = path

        if format is None:
            self.format = path.split(".")[-1]
        else:
            self.format = format

        self.split_seed = split_seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size

    def setup(self, stage=None):
        """Sets up the data for the model.

        Parameters:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Raises:
            NotImplementedError: If the file format is not supported.

        Returns:
            None
        """
        print("Setting up data")

        match self.format:
            case "feather":
                data = pd.read_feather(self.path)
            case "parquet":
                data = pd.read_parquet(self.path)
            case "csv":
                data = pd.read_csv(self.path)
            case _:
                raise NotImplementedError(
                    "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
                )

        self.X = data.drop(["Phenotype", "Condition", "Strain"], axis=1)
        self.y = data["Phenotype"].astype(np.float32)

        try:
            self.condition = data["Condition"]
        except:  # noqa: E722
            self.condition = None

        # Split data into test and training subsets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.split_seed,
        )

        self.test_dataset = EncodeDataset(X_val.values, y_val.values)

        del X_val, y_val

        # Split training subset into training and validation subsets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.test_size,
            random_state=self.split_seed,
        )

        self.train_dataset = EncodeDataset(X_train.values, y_train.values)
        self.val_dataset = EncodeDataset(X_val.values, y_val.values)

        del X_train, X_val, y_train, y_val, self.X, self.y

    def train_dataloader(self):
        """Generates a training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Generates a validation data loader.

        Returns:
            DataLoader: A data loader object.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create and return a DataLoader object for the test dataset.

        Returns:
            A DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )


class CancerDataset(Dataset):
    """Dataset for the Cancer data - PRISM_19Q4."""

    def __init__(
        self,
        X: DataFrameLike,
        y: Union[ArrayLike, pd.Series],
        geno_data: pd.DataFrame,
        latent_data: pd.DataFrame,
    ) -> None:
        self.strain, self.condition = (
            X[X.columns[0]].values,
            X[X.columns[1]].values,
        )  # Contains information on the observations - should be of the form: [cell_line, condition]
        self.y = y  # Phenotype
        self.geno_data = geno_data  # Genotype data - Index should contain the strain information
        self.latent_data = (
            latent_data  # Latent data - Index should contain the condition information
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        strains, conditions = self.strain[idx], self.condition[idx]
        return np.hstack(
            (self.geno_data.loc[strains].values, self.latent_data.loc[conditions].values),
            dtype=np.float32,
        ), self.y[idx].reshape(
            1,
        )


class CancerKFoldModule(LightningDataModule):
    """Lightning data module for the  Cancer Dataset - PRISM_19Q4."""

    def __init__(
        self,
        path: List[str],
        format: str = "parquet",
        k: int = 0,
        split_seed: int = 42,
        num_splits: int = 5,
        num_workers: int = 4,
        batch_size: int = 64,
        # test_size: float = 0.2,
        # stratify: "str" = None,
    ):
        super().__init__()

        self.split_seed = split_seed
        self.num_workers = num_workers
        self.k = k
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.batch_size = batch_size
        # self.test_size = test_size
        # self.stratify = stratify

        self.path = Path(path)
        print(self.path)
        self.format = format  # TODO: add support for other formats

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.save_hyperparameters()

    def setup(self, stage=None):
        """Sets up the data for the model.

        Parameters:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Raises:
            NotImplementedError: If the file format is not supported.

        Returns:
            None
        """
        console.log("Setting up data")

        latent_data = pd.read_parquet(self.path / "latent.parquet").set_index("Condition")
        geno_data = pd.read_parquet(self.path / "genotype.parquet").set_index("cell_lines")

        x_col = ["cell_lines", "Condition"]
        y_col = "Phenotype"

        if stage == "test":
            # console.log("Reading Testing data")
            self.test_df = pq.read_table(self.path / "test.parquet")

            self.test_dataset = CancerDataset(
                self.test_df.to_pandas()[x_col],
                self.test_df.to_pandas()[y_col],
                geno_data=geno_data,
                latent_data=latent_data,
            )
            del self.test_df

        elif stage == "fit":
            # console.log("Reading Training data")
            self.train_df = pl.read_parquet(self.path / "train.parquet")
            # console.log("Read Training data")

            # console.log('Preparing dataset for KFold')
            self.dataset_for_split = CancerDataset(
                self.train_df.to_pandas()[x_col],
                self.train_df.to_pandas()[y_col],
                geno_data=geno_data,
                latent_data=latent_data,
            )
            del self.train_df

            # console.log("Generating KFold splits")
            # KFold validation data generation
            if not self.train_dataset and not self.val_dataset:
                # Create KFold object
                kf = KFold(
                    n_splits=self.num_splits,
                    shuffle=True,
                    random_state=self.split_seed,
                )

                # Generate KFold splits
                all_splits = list(kf.split(self.dataset_for_split))
                train_index, val_index = all_splits[self.k]  # Get indices for particular fold

                # Create training and validation subsets
                self.train_dataset = Subset(self.dataset_for_split, train_index)
                self.val_dataset = Subset(self.dataset_for_split, val_index)

                del self.dataset_for_split

    def train_dataloader(self):
        """Generates a training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Generate a validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Generate a function comment for the given function body in a markdown code block with
        the correct language syntax.

        Returns:
            DataLoader: The DataLoader object created with the specified parameters.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            # num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Testing

    # dm1 = KFoldEncodeModule(
    #     path="/storage/bt20d204/data/bloom2013_shared_clf_3_pubchem.feather",
    # )
    # dm1.setup()

    # dm2 = KFoldEncodeModule(
    #     path="/storage/bt20d204/data/bloom2013_shared_clf_3_pubchem.feather",
    #     k=1,
    #     stratify="Phenotype",
    # )
    # dm2.setup()

    # loader = dm1.val_dataloader()
    # X, y = next(iter(loader))

    # loader2 = dm2.val_dataloader()

    # X2, y2 = next(iter(loader2))
    # print(X, "\n", X2)

    # print(torch.eq(X, X2).all().item()), "X and X2 is different"

    dm3 = CancerKFoldModule(
        path="/home/rajeeva/Project/data/cancer/PRISM_19Q4/", batch_size=256, num_workers=10
    )
    dm3.setup(stage="fit")

    for X, y in dm3.val_dataloader():
        print(X.shape, y.shape, end="\t")
    pass

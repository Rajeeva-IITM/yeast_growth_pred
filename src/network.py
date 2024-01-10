from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    """Neural Network.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hidden_layers: List[int],
        input_size: int,
        output_size: int,
        dropout: float,
        activation: str = "relu",
    ):
        """Neural Network.

        Args:
            hidden_layers (List[int]): Sizes of the hidden layers
            input_size (int): Input size
            output_size (int): Output size
            dropout (float): Dropout
            activation (str): Activation function. Must be one of ["relu", "tanh", "sigmoid", "celu", "gelu"].
        """

        if len(hidden_layers) < 1:
            raise ValueError("Number of layers must be greater than 1.")

        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1.")

        if input_size < 1 or output_size < 1:
            raise ValueError("Input and output sizes must be greater than 1.")

        super().__init__()

        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        match activation:
            case "relu":
                self.activation = nn.ReLU()
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "celu":
                self.activation = nn.CELU()
            case "gelu":
                self.activation = nn.GELU()
            case _:
                self.activation = nn.ReLU()

        self.layers = [self.input_size] + self.hidden_layers + [self.output_size]

        self.sequence = []

        for i in range(len(self.layers) - 2):
            self.sequence.append(nn.Linear(self.layers[i], self.layers[i + 1], dtype=torch.float))
            self.sequence.append(nn.BatchNorm1d(self.layers[i + 1], dtype=torch.float))
            self.sequence.append(self.activation)

        self.sequence.append(nn.Dropout(self.dropout))
        self.sequence.append(nn.Linear(self.layers[-2], self.layers[-1], dtype=torch.float))
        self.sequence = nn.Sequential(*self.sequence)

    def forward(self, x):
        """This function takes in an input tensor `x` and passes it through the neural network's
        sequence of layers. It then returns the output tensor produced by the forward pass.

        Parameters:
        -----------
        x : tensor
            The input tensor to be passed through the neural network.

        Returns:
        --------
        tensor
            The output tensor produced by the forward pass through the network's layers.
        """
        return self.sequence(x)


class MultiViewNet(nn.Module):
    """Neural Network but with two views.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        layers_before_concat: List[int],
        layers_after_concat: List[int],
        input_size: List[int],
        output_size: int,
        dropout: float,
        activation: str = "relu",
    ):
        """_summary_

        Args:
            layers_before_concat (List[int]): Number of layers before the concatenation
            layers_after_concat (List[int]): Number of layers after the concatenation
            input_size (List[int]): The input sizes of two views
            output_size (int): The output size
            dropout (float): Dropout
            activation (str, optional): Activation function. Must be one of ["relu", "tanh", "sigmoid", "celu", "gelu"]. Defaults to "relu".
        """

        if (len(layers_before_concat) < 1) or (len(layers_after_concat) < 1):
            raise ValueError("Number of layers must be greater than 1.")

        if dropout < 0 or dropout > 1:
            raise ValueError("Dropout must be between 0 and 1.")

        if output_size < 1:
            raise ValueError("Output size must be greater than 1.")

        if len(input_size) != 2:
            raise ValueError("Input size must be 2.")

        # if len(layers_after_concat) != len(layers_before_concat):  #TODO: This is only for simplicity's sake
        #     raise ValueError("Number of layers before and after concatenation must be the same.")

        super().__init__()

        self.layers_before_concat = layers_before_concat
        self.layers_after_concat = layers_after_concat
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        match activation:
            case "relu":
                self.activation = nn.ReLU()
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "celu":
                self.activation = nn.CELU()
            case "gelu":
                self.activation = nn.GELU()
            case _:
                raise ValueError("Activation function not supported.")

        # Building the network

        self.view1 = [
            nn.Linear(input_size[0], layers_before_concat[0], dtype=torch.float),
            nn.BatchNorm1d(layers_before_concat[0], dtype=torch.float),
            self.activation,
        ]
        self.view2 = [
            nn.Linear(input_size[1], layers_before_concat[0], dtype=torch.float),
            nn.BatchNorm1d(layers_before_concat[0], dtype=torch.float),
            self.activation,
        ]

        # postconcatenation
        self.postconcat = [
            nn.Linear(layers_before_concat[-1] * 2, layers_after_concat[0], dtype=torch.float),
            nn.BatchNorm1d(layers_after_concat[0], dtype=torch.float),
            self.activation,
        ]

        for i in range(len(self.layers_before_concat) - 2):
            self.view1.append(
                nn.Linear(
                    self.layers_before_concat[i],
                    self.layers_before_concat[i + 1],
                    dtype=torch.float,
                )
            )
            self.view2.append(
                nn.Linear(
                    self.layers_before_concat[i],
                    self.layers_before_concat[i + 1],
                    dtype=torch.float,
                )
            )
            self.view1.append(nn.BatchNorm1d(self.layers_before_concat[i + 1], dtype=torch.float))
            self.view2.append(nn.BatchNorm1d(self.layers_before_concat[i + 1], dtype=torch.float))

            self.view1.append(self.activation)
            self.view2.append(self.activation)

        self.view1.append(nn.Dropout(self.dropout))
        self.view2.append(nn.Dropout(self.dropout))

        self.view1.append(
            nn.Linear(
                self.layers_before_concat[-2], self.layers_before_concat[-1], dtype=torch.float
            )
        )
        self.view2.append(
            nn.Linear(
                self.layers_before_concat[-2], self.layers_before_concat[-1], dtype=torch.float
            )
        )

        self.view1 = nn.Sequential(*self.view1)
        self.view2 = nn.Sequential(*self.view2)

        for i in range(len(self.layers_after_concat) - 2):
            self.postconcat.append(
                nn.Linear(
                    self.layers_after_concat[i], self.layers_after_concat[i + 1], dtype=torch.float
                )
            )
            self.postconcat.append(
                nn.BatchNorm1d(self.layers_after_concat[i + 1], dtype=torch.float)
            )
            self.postconcat.append(self.activation)

        self.postconcat.append(
            nn.Linear(
                self.layers_after_concat[-2], self.layers_after_concat[-1], dtype=torch.float
            )
        )
        self.postconcat.append(nn.Dropout(self.dropout))
        self.postconcat.append(
            nn.Linear(self.layers_after_concat[-1], self.output_size, dtype=torch.float)
        )
        self.postconcat = nn.Sequential(*self.postconcat)

    def forward(self, x):
        x1 = self.view1(x[:, : self.input_size[0]])
        x2 = self.view2(x[:, self.input_size[0] :])

        x = torch.cat((x1, x2), dim=1)
        x = self.postconcat(x)

        return x


if __name__ == "__main__":
    # Testing
    # model = Net(
    #     hidden_layers=[512, 256, 128],
    #     input_size=6000,
    #     output_size=1,
    #     dropout=0.2,
    #     activation= "celu"
    # )

    model = MultiViewNet(
        layers_before_concat=[256, 128, 64, 32, 16, 512],
        layers_after_concat=[2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
        input_size=[600, 600],
        output_size=1,
        dropout=0.2,
    )

    test = torch.randn(20, 1200, dtype=torch.float32)

    print(model)

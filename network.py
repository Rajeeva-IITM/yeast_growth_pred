import numpy as np
import torch
from torch import nn
from typing import Optional, List,  Callable

class Net(nn.Module):
    """Neural Network

    Args:
        nn (_type_): _description_
    """
    
    def __init__(
        self,
        hidden_layers: List[int],
        input_size: int,
        output_size: int,
        dropout: float,
        activation: str = "relu"
    ):
        """Neural Network

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
        
        super(Net, self).__init__()
        
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
            self.sequence.append(
                nn.Linear(self.layers[i], self.layers[i + 1], dtype=torch.float)
            )
            self.sequence.append(
                nn.BatchNorm1d(self.layers[i + 1], dtype=torch.float)
            )
            self.sequence.append(self.activation)
        
        self.sequence.append(nn.Dropout(self.dropout))
        self.sequence.append(
            nn.Linear(self.layers[-2], self.layers[-1], dtype=torch.float)
        )
        self.sequence = nn.Sequential(*self.sequence)
        
    def forward(self, x):
        """
        This function takes in an input tensor `x` and passes it through the neural network's 
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
    
    
if __name__ == "__main__":
    
    # Testing
    model = Net(
        hidden_layers=[512, 256, 128],
        input_size=6000,
        output_size=1,
        dropout=0.2,
        activation= "celu"
    )
    
    test = torch.randn(10, 6000, dtype=torch.float32)
    
    print(model(test))
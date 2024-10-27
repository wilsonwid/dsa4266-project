# Heavily adapted from https://github.com/LixiaoTHU/RCNN/tree/master
import torch
import torch.nn as nn

from utils.types import NonlinearityEnum
from utils.utils import select_nonlinearity

class RecurrentConvolutionalBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False,
            steps: int = 5,
            nonlinearity: NonlinearityEnum = NonlinearityEnum.RELU
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.steps = steps

        # Convolutional layers
        self.conv = nn.Conv2d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Batch normalization layers
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features=output_dim) for _ in range(steps)])

        # Shortcut layer
        self.shortcut = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    
        # Select nonlinearity
        self.nonlinearity = select_nonlinearity(nl_enum=nonlinearity)

        if self.nonlinearity is None:
            raise ValueError("nonlinearity cannot be None or an invalid nonlinearity function!")

        # Initialise the parameters

        for module in self.modules():
            if isinstance(module, nn.Conv2d()):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_x = x
        for i in range(self.steps):
            if i == 0: z = self.conv(x)
            else: z = self.conv(x) + self.shortcut(initial_x)

            x = self.nonlinearity(z)
            x = self.bn[i](x)
        return x

class RecurrentConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

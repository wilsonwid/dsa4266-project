# Heavily adapted from https://github.com/LixiaoTHU/RCNN/tree/master
import torch
import torch.nn as nn

from utils.types import NonlinearityEnum
from utils.utils import select_nonlinearity, compute_linear_size_n

class RecurrentConvolutionalLayer(nn.Module):
    """
    Class representing the recurrent convolutional layer.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = "same",
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
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        collect = []
        for i in range(self.steps):
            if i == 0: z = self.conv(x[:, :, 0, ...].squeeze(2))
            else: z = self.conv(x[:, :, i, ...].squeeze(2)) + self.shortcut(x[:, :, i-1, ...].squeeze(2))

            z = self.nonlinearity(z)
            z = self.bn[i](z)
            collect.append(z)
        return torch.stack(collect, dim=2)

class RecurrentConvolutionalNetwork(nn.Module):
    def __init__(
            self,
            input_channels:  int,
            num_recurrent_layers: int,
            num_kernels: int = 96,
            kernel_size: int | tuple[int, int, int] = 3,
            stride: int | tuple[int, int, int] = 1,
            padding: str | int | tuple[int, int, int] = "same",
            dropout_prob: float = 0.25,
            nonlinearity: NonlinearityEnum = NonlinearityEnum.RELU,
            bias: bool = False,
            steps: int = 5,
            num_classes: int = 2,
        ):
        super().__init__()
        self.input_channels = input_channels
        self.num_recurrent_layers = num_recurrent_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.steps = steps
        self.num_classes = num_classes

        self.nonlinearity_layer = select_nonlinearity(self.nonlinearity)

        self.recurrent_convolutional_layers = nn.ModuleList(
        [RecurrentConvolutionalLayer(
            input_dim=self.input_channels,
            output_dim=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            steps=self.steps,
            nonlinearity=self.nonlinearity
        )] +    
        [RecurrentConvolutionalLayer(
            input_dim=self.num_kernels,
            output_dim=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            steps=self.steps,
            nonlinearity=self.nonlinearity
        ) for _ in range(self.num_recurrent_layers - 1)])


        self.fc1 = nn.Linear(
            in_features=self.num_kernels * self.steps * 224,
            out_features=64,
            bias=self.bias
        )

        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.activation1 = select_nonlinearity(self.nonlinearity)

        self.fc2 = nn.Linear(64, self.num_classes)

        self.softmax = nn.Softmax(dim=1)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        for i in range(0, self.num_recurrent_layers, 2):
            x = self.recurrent_convolutional_layers[i](x)
            if i != self.num_recurrent_layers - 1:
                x = self.recurrent_convolutional_layers[i+1](x)
        
        x = x.mean(dim=4, keepdim=True)
        x = x.flatten(start_dim=1)
        x = nn.functional.dropout(x, p=self.dropout_prob)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

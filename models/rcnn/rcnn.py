# Heavily adapted from https://github.com/LixiaoTHU/RCNN/tree/master
import torch
import torch.nn as nn

from utils.types import NonlinearityEnum
from utils.utils import select_nonlinearity

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
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.size())
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
            num_recurent_layers: int,
            num_kernels: int = 96,
            kernel_size: int | tuple[int, int, int] = 3,
            stride: int | tuple[int, int, int] = 1,
            padding: int | tuple[int, int, int] = 1,
            dropout_prob: float = 0.25,
            nonlinearity: NonlinearityEnum = NonlinearityEnum.RELU,
            bias: bool = False,
            steps: int = 5,
            num_classes: int = 2
        ):
        super().__init__()
        self.input_channels = input_channels
        self.num_recurrent_layers = num_recurent_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.steps = steps
        self.num_classes = num_classes

        self.init_conv_layer = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )

        self.nonlinearity_layer = select_nonlinearity(self.nonlinearity)

        self.batchnorm_layer = nn.BatchNorm3d(num_features=self.num_kernels)

        self.pooling_layers = nn.ModuleList([nn.MaxPool3d(
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        ) for _ in range(self.num_recurrent_layers // 2)])

        self.dropout_layers = nn.ModuleList([nn.Dropout(
            p=self.dropout_prob,
        ) for _ in range(self.num_recurrent_layers // 2)])

        self.recurrent_convolutional_layers = nn.ModuleList([RecurrentConvolutionalLayer(
            input_dim=self.num_kernels,
            output_dim=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            steps=self.steps,
            nonlinearity=self.nonlinearity
        ) for _ in range(self.num_recurrent_layers)])

        self.fc = nn.Linear(
            in_features=self.kernel_size,
            out_features=self.num_classes,
            bias=self.bias
        )

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.init_conv_layer(x)
        x = self.batchnorm_layer(self.nonlinearity_layer(x))
        for i in range(0, self.num_recurrent_layers, 2):
            x = self.recurrent_convolutional_layers[i](x)
            if i != self.num_recurrent_layers - 1:
                x = self.recurrent_convolutional_layers[i+1](x)
            x = self.dropout_layers[i//2](x)
        
        x = nn.functional.max_pool3d(x, kernel_size=self.kernel_size)
        x = x.view(-1, self.num_kernels)
        x = nn.functional.dropout(x, p=self.dropout_prob)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn

from transformers import Vi

class ViViT(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_cnn_layers: int = 5,
            num_kernels: int = 128,
            kernel_size: int | tuple[int, int] = 3,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 1,
            dropout_prob: float = 0.25,
            bias: bool = False,
            num_lstm_layers: int = 5,
            num_classes: int = 2,
            bidirectional: bool = True,
            input_shape: tuple[int, int] = (224, 224),
            fc_size: int = FC_SIZE
        ):
        super().__init__()
        self.input_channels = input_channels
        self.num_cnn_layers = num_cnn_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.input_shape = input_shape
        self.fc_size = fc_size

        self.input_conv = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )

        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels=self.num_kernels,
                      out_channels=self.num_kernels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding,
                      bias=self.bias)
            for _ in range(self.num_cnn_layers)
        ])

        self.poolings = nn.ModuleList([
            nn.MaxPool3d(kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=self.padding)
        for _ in range(self.num_cnn_layers)])

        self.fc = nn.Linear(self.fc_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.input_conv(x)

        for i in range(0, self.num_cnn_layers, 1):
            x = self.convs[i](x)
            x = self.poolings[i](x)

        x = self.softmax(self.fc(x.flatten(start_dim=1)))
        return x



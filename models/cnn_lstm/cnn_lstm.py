import torch
import torch.nn as nn

from utils.types import NonlinearityEnum
from utils.utils import select_nonlinearity

FC_SIZE = 205520896

class CNN_LSTM(nn.Module):
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
            hidden_size: int = 128,
            num_classes: int = 2,
            bidirectional: bool = True,
            input_shape: tuple[int, int] = (224, 224),
            fc_size: int = FC_SIZE,
            seq_len: int = 32
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
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.input_shape = input_shape
        self.fc_size = fc_size
        self.seq_len = seq_len

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

        self.lstm = nn.LSTM(
            input_size=self.num_kernels * self.input_shape[0] * self.input_shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout_prob,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.fc = nn.Linear(2 * self.hidden_size * self.seq_len, self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.input_conv(x)

        for i in range(0, self.num_cnn_layers, 1):
            x = self.convs[i](x)
            x = self.poolings[i](x)

        x = x.permute(0, 2, 1, 3, 4)
        x = x.flatten(start_dim=2)
        (x, _) = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn

from utils.types import NonlinearityEnum
from utils.utils import select_nonlinearity

class CNN_LSTM(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_cnn_layers: int,
            num_kernels: int,
            num_lstm_layers: int,
        ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



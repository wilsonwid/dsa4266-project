import torch
import torch.nn as nn

from utils.utils import compute_linear_size_n, select_nonlinearity
from utils.types import NonlinearityEnum

class ConvBlock(nn.Module):
    """
    Convolutional block with skip connections. Similar to ResNet block.
    """
    def __init__(
            self,
            nonlinearity: NonlinearityEnum = NonlinearityEnum.RELU,
            num_kernels: int = 128,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False
        ):
        super().__init__()

        self.nonlinearity = select_nonlinearity(nonlinearity)
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv1 = nn.Conv2d(
            in_channels=self.num_kernels,
            out_channels=self.num_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.num_kernels)
        self.activation1 = self.nonlinearity

        self.conv2 = nn.Conv2d(
            in_channels=self.num_kernels,
            out_channels=self.num_kernels * 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.num_kernels * 2)

        self.skip_conv = nn.Conv2d(
            in_channels=self.num_kernels,
            out_channels=self.num_kernels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias
        )
        self.last_activation = self.nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x_skip = self.skip_conv(x_skip)        
        x += x_skip
        x = self.last_activation(x)
        return x

class CNN_Section(nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            num_cnn_layers: int = 5,
            num_start_kernels: int = 64,
            kernel_size: int = 3,
            stride: int = 1,
            padding: str = "same",
            dropout_prob: float = 0.25,
            bias: bool = False,
            num_classes: int = 2,
            input_shape: tuple[int, int] = (224, 224),
            steps: int = 16,
            nonlinearity: NonlinearityEnum = NonlinearityEnum.RELU,
            train_alone: bool = True
    ): 
        super().__init__()

        self.input_channels = input_channels
        self.num_cnn_layers = num_cnn_layers
        self.num_start_kernels = num_start_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.steps = steps
        self.nonlinearity = nonlinearity
        self.train_alone = train_alone
        
        self.input_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.num_start_kernels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.bn = nn.BatchNorm2d(num_features=self.num_start_kernels)
        self.activation = select_nonlinearity(self.nonlinearity)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.image_size_after_activation = compute_linear_size_n(
            input_shape=self.input_shape[0], 
            kernel_size=3,
            stride=1,
            padding=1,
            n=1
        )

        self.step_size_after_activation = compute_linear_size_n(
            input_shape=self.steps,
            kernel_size=3,
            stride=1,
            padding=1,
            n=1
        )

        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                nonlinearity=self.nonlinearity,
                num_kernels=self.num_start_kernels * 2 ** i,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias
            )
        for i in range(self.num_cnn_layers)])

        self.image_size_aft_conv = compute_linear_size_n(
            input_shape=self.image_size_after_activation,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            n=2 * self.num_cnn_layers
        )

        self.fc1 = nn.Linear(self.image_size_aft_conv * self.image_size_aft_conv * self.num_start_kernels * 2 ** self.num_cnn_layers, self.image_size_aft_conv ** 2)
        self.activation1 = select_nonlinearity(nonlinearity)
        self.dropout1 = nn.Dropout(self.dropout_prob)

        self.fc2 = nn.Linear(self.image_size_aft_conv**2, self.image_size_aft_conv)
        self.activation2 = select_nonlinearity(nonlinearity)
        self.dropout2 = nn.Dropout(self.dropout_prob)

        self.fc3 = nn.Linear(self.image_size_aft_conv, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.max_pool(x)
        for i in range(self.num_cnn_layers):
            x = self.conv_blocks[i](x)
        
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        if self.train_alone:
            x = self.fc2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
        
        return x
            

class CNN_LSTM_Separate(nn.Module):
    def __init__(
            self,
            cnn_section: CNN_Section,
            num_lstm_layers: int,
            hidden_size: int,
            num_classes: int,
            bidirectional: bool,
            steps: int,
            bias: bool,
            dropout_prob: float,
            nonlinearity: NonlinearityEnum
        ):
        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.steps = steps
        self.bias = bias
        self.dropout_prob = dropout_prob
        self.nonlinearity = nonlinearity

        self.cnn_section = cnn_section
        for param in self.cnn_section.parameters():
            param.requires_grad = False

        self.cnn_section.train_alone = False

        # input size of lstm will be the output of the last pooling layer
        # output of the last pooling layer will have a size of (batch_size, num_kernels, steps, seq_len, seq_len)
        # we switch it to make it (batch_size, steps, num_kernels, seq_len, seq_len)
        # and then flatten it to make it (batch_size, steps, num_kernels * seq_len * seq_len)
        self.lstm = nn.LSTM(
            input_size=self.cnn_section.image_size_aft_conv ** 2, 
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout_prob,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        # output of lstm will have a size of (batch_size, steps, (2 if bidirectional else 1) * hidden_size)
        # we flatten it to make it (batch_size, steps * (2 if bidirectional else 1) * hidden_size)
        self.fc1 = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size * self.steps, 64)
        self.activation1 = select_nonlinearity(self.nonlinearity)
        self.dropout1 = nn.Dropout(self.dropout_prob)

        self.fc2 = nn.Linear(64, self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        collected = []
        for t in range(x.size(2)):
            cur = x[:, :, t, :, :]
            cur = self.cnn_section(cur)
            collected.append(cur)

        x = torch.stack(collected, dim=2)

        # now x will be of dimension (batch_size, steps, self.cnn_section.image_size_aft_conv ** 2)

        (x, _) = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

import torch
import torch.nn as nn

from transformers import ViTMAEModel

class CustomViTMAE(nn.Module):
    def __init__(
            self,
            dropout_prob: float = 0.25,
            num_classes: int = 2,
            in_channels: int = 3,
            kernel_size: int = 3,
            steps: int = 32
        ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.steps = steps

        self.video_conv = nn.Conv3d(in_channels=self.in_channels, out_channels=1, kernel_size=self.kernel_size, padding=1)
        self.proc_conv = nn.Conv3d(in_channels=self.in_channels, out_channels=1, kernel_size=self.kernel_size, padding=1)

        self.video_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
        self.proc_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")

        self.video_conv2d = nn.Conv2d(in_channels=self.steps,
                                      out_channels=3,
                                      kernel_size=self.kernel_size,
                                      padding=1)
        self.proc_conv2d = nn.Conv2d(in_channels=self.steps,
                                     out_channels=3,
                                     kernel_size=self.kernel_size, padding=1)

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc = nn.Linear(self.video_model.config.hidden_size, self.num_classes)

    def forward(self, **inputs) -> torch.Tensor:
        video_in, proc_in = inputs["video"].permute(0, 2, 1, 3, 4), inputs["proc"].permute(0, 2, 1, 3, 4)

        video_in = self.video_conv(video_in).squeeze(1)
        proc_in = self.proc_conv(proc_in).squeeze(1)

        img_out = self.video_conv2d(video_in)
        proc_out = self.proc_conv2d(proc_in)

        img_out = self.video_model(img_out)
        proc_out = self.proc_model(proc_out)

        combined = torch.concat([img_out.last_hidden_state, proc_out.last_hidden_state], dim=1)
        combined = combined.mean(dim=1)

        combined = self.dropout(combined)
        combined = self.fc(combined)

        return combined

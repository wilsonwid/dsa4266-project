import torch
import torch.nn as nn

from transformers import ViTMAEModel

class CustomViTMAE(nn.Module):
    def __init__(
            self,
            dropout_prob: float = 0.25,
            num_classes: int = 2
        ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.video_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
        self.proc_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc = nn.Linear(self.video_model.config.hidden_size, self.num_classes)

    def forward(self, **inputs) -> torch.Tensor:
        img_out = self.video_model(inputs["video"])
        proc_out = self.proc_model(inputs["proc"])

        combined = torch.stack([img_out, proc_out], dim=1).mean(dim=1)

        combined = self.dropout(combined)
        combined = self.fc(combined)

        return combined

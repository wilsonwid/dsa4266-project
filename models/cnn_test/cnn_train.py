import torch
import torchvision
import torch.nn as nn
import os
import sys
import numpy as np

from cnn_test import CNNModel
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torchvision.transforms import v2

BATCH_SIZE = 128
LR = 1e-3

main_folder_path = os.path.dirname(__file__) + "/../.."
sys.path.append(main_folder_path)

transforms = v2.Compose([
    v2.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(f"{main_folder_path}/AUGMENTED/train_frames_augmented", transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = torchvision.datasets.ImageFolder(f"{main_folder_path}/AUGMENTED/validation_frames_augmented", transform=transforms)

loss_fn = nn.BCELoss()
model = CNNModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


for epoch in tqdm(range(5)):
    print(f"Epoch: {epoch}")
    model.train()
    collected_labels, collected_predictions = [], []
    running_loss = 0.
    for idx, data in tqdm(enumerate(train_loader)):
        img, labels = data
        optimizer.zero_grad()
        output = model(img)

        labels = labels.unsqueeze(dim=1).type(torch.float32)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        numpy_labels = labels.numpy().astype(np.uint8).tolist()
        actual_predictions = output.detach().numpy().round().astype(np.uint8).tolist()

        collected_labels.extend(numpy_labels)
        collected_predictions.extend(actual_predictions)

        print(labels, output)

        if idx % 10 == 9:
            last_loss = running_loss / 10
            print(last_loss)
            collected_labels = np.array(collected_labels)
            collected_predictions = np.array(collected_predictions)

            epoch_f1 = f1_score(collected_labels, collected_predictions, average="weighted")
            epoch_acc = accuracy_score(collected_labels, collected_predictions)

            print(collected_labels, collected_predictions)

            print(f"Batch: {idx + 1}, Loss: {last_loss}, F1 score: {epoch_f1}, Accuracy score: {epoch_acc}")
            # tb_x = epoch * len(train_loader) + idx + 1

            running_loss = 0.

            collected_labels, collected_predictions = [], []


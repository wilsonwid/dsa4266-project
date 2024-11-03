# Heavily adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import sys
import os
main_folder_path = os.path.dirname(__file__) + "/../.."
sys.path.append(main_folder_path)
import torch
import torch.nn as nn
import datetime as dt
import argparse
import subprocess

from utils.dataset import VideoDataset
from models.vitmae.vitmae import CustomViTMAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "vitmae"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime('%Y-%m-%d-%H-%M-%S')}"
SAVE_DIR = f"{main_folder_path}/models/vitmae/saved_models"
DATA_FOLDER = "data"
INF = 100000000.

timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/rcnn_{timestamp}")

def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    epochs: int,
    filename: str,
    writer: SummaryWriter
):
    """
    Trains the model and saves the weights into a `.pt` file.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Data to train the model on.
        val_dataloader (torch.utils.data.DataLoader): Data to validate the model on.
        model (nn.Module): Model to be trained.
        epochs (int): Number of epochs.
        filename (str): Filename to save the model to.

    Returns:

    """
    print(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.
    last_loss = 0.
    best_vloss = INF

    for epoch in tqdm(range(epochs)):
        model.train()
        collected_labels, collected_predictions = [], []
        for i, data in tqdm(enumerate(train_dataloader)):
            subprocess.run(["nvidia-smi"])
            vid_inputs, proc_inputs, labels = data["video"].to(device), data["proc"].to(device), data["target"].to(device)

            optimizer.zero_grad()
            output = model(video=vid_inputs, proc=proc_inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            collected_labels.append(labels.cpu())
            collected_predictions.append(output.argmax(dim=1).cpu())

            if i % 10 == 9:
                last_loss = running_loss / 10
                epoch_f1 = f1_score(torch.cat(collected_labels), torch.cat(collected_predictions), average="weighted")
                print(f"Batch: {i + 1}, Loss: {last_loss}, F1 score: {epoch_f1}")
                tb_x = epoch * len(train_dataloader) + i + 1

                writer.add_scalar("Loss/train", last_loss, tb_x)
                writer.add_scalar("F1 score/train", epoch_f1, tb_x)

                running_loss = 0.

        model.eval()
        with torch.no_grad():
            collected_labels, collected_predictions = [], []
            for i, vdata in enumerate(val_dataloader):
                vid_inputs, proc_inputs, labels = vdata["video"].to(device), vdata["proc"].to(device), vdata["target"].to(device)
                output = model(video=vid_inputs, proc=proc_inputs)
                loss = loss_fn(output, labels)
                running_loss += loss.item()

                collected_labels.append(labels.cpu())
                collected_predictions.append(output.argmax(dim=1).cpu())
            val_f1 = f1_score(torch.cat(collected_labels), torch.cat(collected_predictions), average="weighted")
            print(f"Validation Loss: {loss.item()}, Validation F1 score: {val_f1}")


        avg_vloss = running_loss / (i + 1)
        print(f"Train Loss: {last_loss}, Val Loss: {avg_vloss}")

        writer.add_scalars("Training vs Validation Loss",
                           {"Train": last_loss, "Validation": avg_vloss},
                           epoch + 1)
        writer.add_scalars("Training vs Validation F1 Score",
                           {"Train": epoch_f1, "Validation": val_f1},
                           epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"{SAVE_DIR}/{MODEL_NAME}_{timestamp}_{epoch + 1}.pt"

            torch.save(model.state_dict(), model_path)


    torch.save(model.state_dict(), f"{SAVE_DIR}/{MODEL_NAME}_{filename}.pt")



def get_arguments() -> argparse.Namespace:
    """
    Parses the arguments for the training script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="train_rcnn",
        description="Trains the RCNN Model"
    )
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of steps")
    parser.add_argument("--dropout_prob", type=float, default=0.25,
                        help="Dropout probability")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--filename", type=str, default=FILENAME,
                        help="Filename to save the model")
    parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for training")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_arguments()
    model = CustomViTMAE(
        dropout_prob=args.dropout_prob,
        num_classes=args.num_classes
    )

    model = nn.DataParallel(model)

    train_dataset = VideoDataset(root=f"{main_folder_path}/data/train", clip_len=args.steps)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8)

    val_dataset = VideoDataset(root=f"{main_folder_path}/data/validation", clip_len=args.steps)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8)
    
    train_model(
        train_dataloader=train_loader, 
        val_dataloader=val_loader,
        model=model, 
        epochs=args.epochs, 
        filename=args.filename, 
        writer=writer
    )

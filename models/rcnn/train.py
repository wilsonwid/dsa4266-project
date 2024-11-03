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

from rcnn import RecurrentConvolutionalNetwork
from utils.dataset import VideoDataset
from utils.utils import convert_str_to_nonlinearity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "rcnn"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime('%Y-%m-%d-%H-%M-%S')}"
SAVE_DIR = f"{main_folder_path}/models/rcnn/saved_models"
DATA_FOLDER = "data"
INF = 100000000.
FC_SIZE = 7009280

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
            inputs, labels = data["video"].to(device), data["target"].to(device)

            optimizer.zero_grad()
            output = model(inputs)
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
                inputs, labels = vdata["video"].to(device), vdata["target"].to(device)
                output = model(inputs)
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
    parser.add_argument("--input_dim", type=int, default=6,
                        help="Input dimension")
    parser.add_argument("--num_rec_layers", type=int, default=5, 
                        help="Number of recurrent layers")
    parser.add_argument("--num_kernels", type=int, default=128,
                        help="Number of kernels in each RCL layer")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Kernel size (as int)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride (as int)")
    parser.add_argument("--padding", type=int, default=1,
                        help="Padding (as int)")
    parser.add_argument("--dropout_prob", type=float, default=0.25,
                        help="Dropout probability")
    parser.add_argument("--nonlinearity", type=str, default="relu",
                        help="Nonlinearity function (as str)")
    parser.add_argument("--bias", type=bool, default=False,
                        help="Bias (as bool)")
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of steps")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--filename", type=str, default=FILENAME,
                        help="Filename to save the model")
    parser.add_argument("--batch_size", type=int, default=12,
                    help="Batch size for training")
    parser.add_argument("--fc_size", type=int, default=FC_SIZE,
                        help="Last fully connected layer size")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_arguments()
    model = RecurrentConvolutionalNetwork(
        input_channels=args.input_dim,
        num_recurent_layers=args.num_rec_layers,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        dropout_prob=args.dropout_prob,
        nonlinearity=convert_str_to_nonlinearity(args.nonlinearity),
        bias=args.bias,
        steps=args.steps,
        num_classes=args.num_classes,
        fc_size=args.fc_size
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

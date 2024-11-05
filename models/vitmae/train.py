# Heavily adapted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html and https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

import sys
import os
main_folder_path = os.path.dirname(__file__) + "/../.."
sys.path.append(main_folder_path)
import torch
import torch.nn as nn
import datetime as dt
import argparse
import subprocess
import ray.cloudpickle as pickle
import numpy as np
import tempfile

from utils.dataset_vitmae import VideoDatasetMAE
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from models.vitmae.vitmae import CustomViTMAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from pathlib import Path
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "vitmae"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime('%Y-%m-%d-%H-%M-%S')}"
SAVE_DIR = f"{main_folder_path}/models/vitmae/saved_models"
DATA_FOLDER = "data"
INF = 100000000.
NUM_WORKERS = 8
NUM_CLASSES = 2
GPUS_PER_TRIAL = torch.cuda.device_count() if torch.cuda.is_available() else 0

timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/rcnn_{timestamp}")

def train_model(
    config: dict,
    epochs: int,
):
    """
    Trains the model and saves the weights into a `.pt` file.

    Args:
        epochs (int): Number of epochs.
        filename (str): Filename to save the model to.
        writer (SummaryWriter): Writer for logs.
        config (dict): Ray Tune dictionary.

    Returns:
        None
    """
    writer = SummaryWriter(f"runs/vitmae_{timestamp}")
    train_dataset = VideoDatasetMAE(
        root=f"{main_folder_path}/data/train", 
        clip_len=config["steps"]
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    val_dataset = VideoDatasetMAE(
        root=f"{main_folder_path}/data/validation", 
        clip_len=config["steps"]
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    model = CustomViTMAE(
        dropout_prob=float(config["dropout_prob"]),
        num_classes=NUM_CLASSES,
        in_channels=3,
        kernel_size=int(config["kernel_size"]),
        stride=int(config["stride"]),
        steps=int(config["steps"])
    )

    print(device)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    running_loss = 0.
    last_loss = 0.
    best_vloss = INF

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    print(f"Starting training at epoch: {start_epoch}")
    for epoch in tqdm(range(start_epoch, epochs)):
        print(f"Epoch: {epoch}")
        model.train()
        collected_labels, collected_predictions = [], []
        for i, data in tqdm(enumerate(train_loader)):
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
                tb_x = epoch * len(train_loader) + i + 1

                writer.add_scalar("Loss/train", last_loss, tb_x)
                writer.add_scalar("F1 score/train", epoch_f1, tb_x)

                running_loss = 0.

        model.eval()
        with torch.no_grad():
            collected_labels, collected_predictions = [], []
            for i, vdata in enumerate(val_loader):
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


        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb")as fp:
                pickle.dump(checkpoint_data, fp)
            
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report({
                "loss": avg_vloss,
                "f1": val_f1
            }, checkpoint=checkpoint)
    
    print("Finished training")

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
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of steps")
    parser.add_argument("--dropout_prob", type=float, default=0.25,
                        help="Dropout probability")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--filename", type=str, default=FILENAME,
                        help="Filename to save the model")
    parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for training")

    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_arguments()
    config = {
        "dropout_prob": tune.choice(np.arange(start=0., stop=1., step=.05).tolist()),
        "kernel_size": tune.choice([3 * i for i in range(1, 11)]),
        "stride": tune.choice([i for i in range(1, 3)]),
        "steps": tune.choice([4 * i for i in range(1, 33)]),
        "batch_size": tune.choice([4, 8, 16, 32, 64, 128]),
        "lr": tune.choice([1e-3, 1e-4, 1e-5, 1e-6])
    }

    gpus_per_trial = GPUS_PER_TRIAL

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2
    )

    print("Running Ray Tune...")

    result = tune.run(
        partial(
            train_model, 
            epochs=args.epochs,
        ),
        resources_per_trial={"cpu": os.cpu_count(), "gpu": gpus_per_trial / 3},
        config=config,
        num_samples=1,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation f1: {best_trial.last_result['f1']}")

    best_trained_model = CustomViTMAE(
        dropout_prob=best_trial.config["dropout_prob"],
        num_classes=2,
        in_channels=3,
        kernel_size=best_trial.config["kernel_size"],
        stride=best_trial.config["stride"],
        padding=1,
        steps=best_trial.config["steps"]
    )

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="f1", mode="max")

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])


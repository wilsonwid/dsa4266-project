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
import pandas as pd

from utils.dataset import VideoDataset
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from models.rcnn.rcnn import RecurrentConvolutionalNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from functools import partial
from utils.types import NonlinearityEnum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "rcnn"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime('%Y-%m-%d-%H-%M-%S')}"
SAVE_DIR = f"{main_folder_path}/models/rcnn/saved_models"
DATA_FOLDER = "data"
INF = 100000000.
NUM_WORKERS = 8
NUM_CLASSES = 2
GPUS_PER_TRIAL = torch.cuda.device_count() if torch.cuda.is_available() else 0

timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    writer = SummaryWriter(f"runs/rcnn_{timestamp}")

    train_dataset = VideoDataset(
        root=f"{main_folder_path}/data/train", 
        clip_len=config["steps"],
        include_additional_transforms=config["include_additional_transforms"]
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    val_dataset = VideoDataset(
        root=f"{main_folder_path}/data/validation", 
        clip_len=config["steps"],
        include_additional_transforms=config["include_additional_transforms"]
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    model = RecurrentConvolutionalNetwork(
        input_channels=config["input_channels"],
        num_recurrent_layers=config["num_recurrent_layers"],
        num_kernels=config["num_kernels"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
        dropout_prob=config["dropout_prob"],
        nonlinearity=config["nonlinearity"],
        bias=config["bias"],
        steps=config["steps"],
        num_classes=config["num_classes"],
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
            vid_inputs, labels = data["video"].to(device), data["target"].to(device)

            optimizer.zero_grad()
            output = model(vid_inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            numpy_labels = labels.cpu().numpy().tolist()
            actual_predictions = output.argmax(dim=1).cpu().numpy().tolist()

            collected_labels.extend(numpy_labels)
            collected_predictions.extend(actual_predictions)

            print(labels, output)

            if i % 10 == 9:
                last_loss = running_loss / 10
                collected_labels = np.array(collected_labels)
                collected_predictions = np.array(collected_predictions)

                epoch_f1 = f1_score(collected_labels, collected_predictions, average="weighted")
                epoch_acc = accuracy_score(collected_labels, collected_predictions)

                print(collected_labels, collected_predictions)

                print(f"Batch: {i + 1}, Loss: {last_loss}, F1 score: {epoch_f1}, Accuracy score: {epoch_acc}")
                tb_x = epoch * len(train_loader) + i + 1

                writer.add_scalar("Loss/train", last_loss, tb_x)
                writer.add_scalar("F1 score/train", epoch_f1, tb_x)
                writer.add_scalar("Accuracy score/train", epoch_acc, tb_x)

                running_loss = 0.

                collected_labels, collected_predictions = [], []

        model.eval()
        with torch.no_grad():
            collected_labels, collected_predictions = [], []
            for i, vdata in enumerate(val_loader):
                vid_inputs, labels = vdata["video"].to(device), vdata["target"].to(device)
                output = model(vid_inputs)
                loss = loss_fn(output, labels)
                running_loss += loss.item()

                numpy_labels = labels.cpu().numpy().tolist()
                actual_predictions = output.argmax(dim=1).cpu().numpy().tolist()

                collected_labels.extend(numpy_labels)
                collected_predictions.extend(actual_predictions)

            collected_labels = np.array(collected_labels)
            collected_predictions = np.array(collected_predictions)

            val_f1 = f1_score(collected_labels, collected_predictions, average="weighted")
            val_acc = accuracy_score(collected_labels, collected_predictions)

            print(f"Validation Loss: {loss.item()}, Validation F1 score: {val_f1}, Validation Accuracy score: {val_acc}")
            print(collected_labels, collected_predictions)

        avg_vloss = running_loss / (i + 1)
        print(f"Train Loss: {last_loss}, Val Loss: {avg_vloss}")

        writer.add_scalars("Training vs Validation Loss",
                           {"Train": last_loss, "Validation": avg_vloss},
                           epoch + 1)
        writer.add_scalars("Training vs Validation F1 Score",
                           {"Train": epoch_f1, "Validation": val_f1},
                           epoch + 1)

        writer.add_scalars("Training vs Validation Accuracy Score",
                           {"Train": epoch_acc, "Validation": val_acc},
                           epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss


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
                "f1": val_f1,
                "acc": val_acc
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
    search_space = {
        "input_channels": 3,
        "num_recurrent_layers": tune.randint(1, 6),
        "num_kernels": tune.randint(1, 64),
        "kernel_size": tune.randint(1, 10),
        "stride": 1,
        "padding": "same",
        "dropout_prob": tune.uniform(0.1, 0.6),
        "nonlinearity": tune.choice([NonlinearityEnum.SILU, NonlinearityEnum.RELU, NonlinearityEnum.GELU, NonlinearityEnum.ELU, NonlinearityEnum.LRELU]),
        "bias": False,
        "steps": tune.choice([2 ** i for i in range(4, 7)]),
        "num_classes": NUM_CLASSES,
        "batch_size": tune.choice([1, 2]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "include_additional_transforms": False
    }

    gpus_per_trial = GPUS_PER_TRIAL

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="acc",
        mode="max",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2
    )

    print("Running Ray Tune...")

    result = tune.run(
        partial(
            train_model, 
            epochs=3
        ),
        resources_per_trial={"cpu": os.cpu_count(), "gpu": gpus_per_trial},
        config=search_space,
        num_samples=20,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("acc", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation f1: {best_trial.last_result['f1']}")
    print(f"Best trial final validation acc: {best_trial.last_result['acc']}")

    best_trained_model = RecurrentConvolutionalNetwork(
        input_channels=best_trial.config["input_channels"],
        num_recurrent_layers=best_trial.config["num_recurrent_layers"],
        num_kernels=best_trial.config["num_kernels"],
        kernel_size=best_trial.config["kernel_size"],
        stride=best_trial.config["stride"],
        padding=best_trial.config["padding"],
        dropout_prob=best_trial.config["dropout_prob"],
        nonlinearity=best_trial.config["nonlinearity"],
        bias=best_trial.config["bias"],
        steps=best_trial.config["steps"],
        num_classes=best_trial.config["num_classes"],
    )

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="acc", mode="max")

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    new_tuner = tune.Tuner(
        partial(
            train_model, 
            epochs=10
        ),
        param_space=best_trial.config,
        tune_config=tune.TuneConfig(
            num_workers=1,
            num_samples=1, 
            use_gpu=True,
            resources_per_worker={
                "CPU": os.cpu_count(),
                "GPU": gpus_per_trial
            }
        )
    )

    new_results = new_tuner.fit()

    dfs = {result.path: result.metrics_dataframe for result in new_results}

    for value in dfs.values():
        value.to_csv(f"{main_folder_path}/models/rcnn/best_result.csv", index=False)

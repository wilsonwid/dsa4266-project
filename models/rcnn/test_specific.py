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
from torchvision.transforms import v2

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
    include_validation: bool
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

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(),
        v2.GaussianBlur(kernel_size=3),
        v2.RandomAdjustSharpness(1.5)
    ])

    train_dataset = VideoDataset(
        root=f"{main_folder_path}/data/train", 
        clip_len=config["steps"],
        include_additional_transforms=config["include_additional_transforms"],
        random_transforms=transforms
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    val_dataset = VideoDataset(
        root=f"{main_folder_path}/data/validation", 
        clip_len=config["steps"],
        include_additional_transforms=config["include_additional_transforms"],
        random_transforms=transforms
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
        
        if include_validation:
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

        else:
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
    search_space = {
        "input_channels": 3,
        "num_recurrent_layers": 2,
        "num_kernels": 60,
        "kernel_size": 9,
        "stride": 1,
        "padding": "same",
        "dropout_prob": 0.185931,
        "nonlinearity": NonlinearityEnum.ELU,
        "bias": False,
        "steps": 64,
        "num_classes": NUM_CLASSES,
        "batch_size": 2,
        "lr": 0.0001236517,
        "include_additional_transforms": False
    }

    gpus_per_trial = GPUS_PER_TRIAL

    best_trained_model = RecurrentConvolutionalNetwork(
        input_channels=search_space["input_channels"],
        num_recurrent_layers=search_space["num_recurrent_layers"],
        num_kernels=search_space["num_kernels"],
        kernel_size=search_space["kernel_size"],
        stride=search_space["stride"],
        padding=search_space["padding"],
        dropout_prob=search_space["dropout_prob"],
        nonlinearity=search_space["nonlinearity"],
        bias=search_space["bias"],
        steps=search_space["steps"],
        num_classes=search_space["num_classes"],
    )

    weights_dir = "/home/w/wilsonwi/ray_results/train_model_2024-11-13_14-55-55/train_model_54b78_00000_0_2024-11-13_14-55-56/checkpoint_000005"

    with open(f"{weights_dir}/data.pkl", "rb") as f:
        result = pickle.load(f)
        best_trained_model.load_state_dict(result["net_state_dict"])

    test_dataset = VideoDataset(
        root=f"{main_folder_path}/data/test",
        epoch_size=None,
    )

    test_dataset = VideoDataset(
        root=f"{main_folder_path}/data/test", 
        clip_len=search_space["steps"],
        include_additional_transforms=search_space["include_additional_transforms"],
        random_transforms=None
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=int(search_space["batch_size"]),
        num_workers=NUM_WORKERS
    )

    collected_fnames = [os.path.basename(output["path"])[:-4] for output in test_dataset]

    collected_labels, collected_predictions = [], []
    probs = []

    best_trained_model = best_trained_model.to(device)

    for i, data in enumerate(test_loader):
        vid_inputs, labels = data["video"].to(device), data["target"].to(device)
        labels = labels.type(torch.float32).unsqueeze(dim=1)
        output = best_trained_model(vid_inputs)

        numpy_labels = labels.cpu().numpy().flatten().tolist()
        actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).flatten().tolist()
        prob = output.detach().cpu().numpy().flatten().tolist()

        collected_labels.extend(numpy_labels)
        collected_predictions.extend(actual_predictions)
        probs.extend(prob)

    df = pd.DataFrame({
        "filename": collected_fnames,
        "actual": collected_labels,
        "predicted": collected_predictions,
        "probability": probs
    })

    df["probability"] = df["probability"].apply(lambda x: 1 - x if x < 0.5 else x) 

    df.to_csv(f"{main_folder_path}/models/{MODEL_NAME}/results/predictions_{MODEL_NAME}.csv")

    print("Finished entire training regime")


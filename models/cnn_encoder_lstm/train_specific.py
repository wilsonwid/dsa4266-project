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
import ray

import numpy as np
import tempfile
import pandas as pd

from ray import tune, train
from utils.dataset import VideoDataset
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from models.cnn_split_lstm.cnn_split_lstm import CNN_Section, CNN_LSTM_Separate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from functools import partial
from utils.types import NonlinearityEnum
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "cnn_split_lstm"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime('%Y-%m-%d-%H-%M-%S')}"
SAVE_DIR = f"{main_folder_path}/models/cnn_split_lstm/saved_models"
DATA_FOLDER = "data"
INF = 100000000.
NUM_WORKERS = 8
NUM_CLASSES = 2
GPUS_PER_TRIAL = torch.cuda.device_count() if torch.cuda.is_available() else 0

timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/cnn_split_lstm_{timestamp}")

def train_cnn_model(
    config: dict,
    epochs: int,
    include_validation: bool,
    trained_model: CNN_Section = None
):
    writer = SummaryWriter(f"runs/cnn_split_lstm_{timestamp}")

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(),
        v2.GaussianBlur(kernel_size=3),
        v2.RandomAdjustSharpness(1.5),
        v2.ToTensor()
    ])

    to_tensor = v2.Compose([
        v2.ToTensor()
    ])



    train_dataset = ImageFolder(f"{main_folder_path}/AUGMENTED/train_frames_augmented", transform=transforms)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=int(config["batch_size"]),
        num_workers=NUM_WORKERS,
        shuffle=True
    )

    val_dataset = ImageFolder(f"{main_folder_path}/AUGMENTED/validation_frames_augmented", transform=to_tensor)

    if include_validation:
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=int(config["batch_size"]),
            num_workers=NUM_WORKERS,
            shuffle=True
        )
    else:
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=int(config["batch_size"]),
            num_workers=NUM_WORKERS,
        )   

    if not trained_model:
        model = CNN_Section(
            input_channels=int(config["input_channels"]),
            num_cnn_layers=int(config["num_cnn_layers"]),
            num_start_kernels=int(config["num_start_kernels"]),
            kernel_size=int(config["kernel_size"]),
            stride=int(config["stride"]),
            padding=int(config["padding"]),
            dropout_prob=float(config["dropout_prob"]),
            bias=False,
            num_classes=NUM_CLASSES,
            input_shape=(224, 224),
            steps=int(config["steps"]),
            nonlinearity=config["nonlinearity"]
        )
    else:
        model = trained_model

    print(device)
    model = model.to(device)

    loss_fn = nn.BCELoss()
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
            # subprocess.run(["nvidia-smi"])
            vid_inputs, labels = data 
            vid_inputs, labels = vid_inputs.to(device), labels.to(device)
            labels = labels.type(torch.float32).unsqueeze(dim=1)

            optimizer.zero_grad()
            output = model(vid_inputs)

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            numpy_labels = labels.detach().cpu().numpy().tolist()
            actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()

            collected_labels.extend(numpy_labels)
            collected_predictions.extend(actual_predictions)

            if i % 10 == 9:
                last_loss = running_loss / 10
                collected_labels = np.array(collected_labels)
                collected_predictions = np.array(collected_predictions)

                epoch_f1 = f1_score(collected_labels, collected_predictions, average="weighted")
                epoch_acc = accuracy_score(collected_labels, collected_predictions)

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
                vid_inputs, labels = vdata 
                labels = labels.type(torch.float32).unsqueeze(dim=1)

                vid_inputs, labels = vid_inputs.to(device), labels.to(device)
                output = model(vid_inputs)
                loss = loss_fn(output, labels)
                running_loss += loss.item()

                numpy_labels = labels.cpu().numpy().tolist()
                actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()

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
                    vid_inputs, labels = vdata 
                    labels = labels.type(torch.float32).unsqueeze(dim=1)
                    vid_inputs, labels = vid_inputs.to(device), labels.to(device)
                    output = model(vid_inputs)
                    loss = loss_fn(output, labels)
                    running_loss += loss.item()

                    numpy_labels = labels.cpu().numpy().tolist()
                    actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()

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
                "val_loss": avg_vloss,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_loss": last_loss,
                "train_f1": epoch_f1,
                "train_acc": epoch_acc
            }, checkpoint=checkpoint)
    
    print("CNN Section finished training") 

def train_model(
    config: dict,
    epochs: int,
    include_validation: bool,
    cnn_model: CNN_Section,
    trained_model: CNN_LSTM_Separate = None
):
    """
    Trains the model and saves the weights into a `.pt` file.

    Args:
        config (dict): Ray Tune dictionary.
        epochs (int): Number of epochs to train the model.
        include_validation (bool): Whether to include validation as training or not.

    Returns:
        None
    """
    cnn_model = ray.get(cnn_model)
    writer = SummaryWriter(f"runs/cnn_lstm_{timestamp}")

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
        num_workers=NUM_WORKERS,
    )

    if not trained_model:
        model = CNN_LSTM_Separate(
            cnn_section=cnn_model,
            num_lstm_layers=int(config["num_lstm_layers"]),
            hidden_size=int(config["hidden_size"]),
            num_classes=2,
            bidirectional=config["bidirectional"],
            steps=int(config["steps"]),
            bias=False,
            dropout_prob=config["dropout_prob"],
            nonlinearity=config["nonlinearity"]
        )
    else:
        model = trained_model

    print(device)
    model = model.to(device)

    loss_fn = nn.BCELoss()
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
            # subprocess.run(["nvidia-smi"])
            vid_inputs, labels = data["video"].to(device), data["target"].to(device)
            labels = labels.type(torch.float32).unsqueeze(dim=1)
            optimizer.zero_grad()
            output = model(vid_inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            numpy_labels = labels.cpu().numpy().tolist()
            actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()

            collected_labels.extend(numpy_labels)
            collected_predictions.extend(actual_predictions)

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
                labels = labels.type(torch.float32).unsqueeze(dim=1)
                optimizer.zero_grad()
                output = model(vid_inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                numpy_labels = labels.cpu().numpy().tolist()
                actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()

                collected_labels.extend(numpy_labels)
                collected_predictions.extend(actual_predictions)

            collected_labels = np.array(collected_labels)
            collected_predictions = np.array(collected_predictions)

            val_f1 = f1_score(collected_labels, collected_predictions, average="weighted")
            val_acc = accuracy_score(collected_labels, collected_predictions)

            print(f"Validation Loss: {loss.item()}, Validation F1 score: {val_f1}, Validation Accuracy score: {val_acc}")
        else:
            model.eval()
            with torch.no_grad():
                collected_labels, collected_predictions = [], []
                for i, vdata in enumerate(val_loader):
                    vid_inputs, labels = vdata["video"].to(device), vdata["target"].to(device)
                    labels = labels.type(torch.float32).unsqueeze(dim=1)
                    output = model(vid_inputs)
                    loss = loss_fn(output, labels)
                    running_loss += loss.item()

                    numpy_labels = labels.cpu().numpy().tolist()
                    actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()    

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
                "val_loss": avg_vloss,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_loss": last_loss,
                "train_f1": epoch_f1,
                "train_acc": epoch_acc
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
        prog="train_cnn_lstm",
        description="Trains the CNN-LSTM Model"
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
    cnn_search_space = {
        "input_channels": 3,
        "num_cnn_layers": 3,
        "num_start_kernels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 5,
        "dropout_prob": 0.05,
        "bias": False,
        "num_classes": NUM_CLASSES,
        "input_shape": (224, 224),
        "batch_size": 64,
        "lr": 0.00137748,
        "steps": 120,
        "nonlinearity": NonlinearityEnum.SILU,
        "include_additional_transforms": False,
    }

    cur_model = CNN_Section(
        input_channels=3,
        num_cnn_layers=3, 
        num_start_kernels=16,
        kernel_size=4,
        stride=2,
        padding=5,
        dropout_prob=0.05,
        bias=False,
        num_classes=NUM_CLASSES,
        input_shape=(224, 224),
        nonlinearity=NonlinearityEnum.SILU,
        train_alone=False
    )

    gpus_per_trial = GPUS_PER_TRIAL

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_acc",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    directory = "/home/wilsonwid/ray_results/train_cnn_model_2024-11-13_00-32-51/train_cnn_model_c2cf0_00002_2_dropout_prob=0.0500,kernel_size=4,lr=0.0001,nonlinearity=ref_ph_2fdecf16,num_cnn_layers=3,num_start__2024-11-13_00-32-51/checkpoint_000001"

    with open(f"{directory}/data.pkl", "rb") as f:
        checkpoint_data = pickle.load(f)
        cur_model.load_state_dict(checkpoint_data["net_state_dict"])

    lstm_search_space = {
        "num_lstm_layers": tune.choice([i for i in range(3, 6)]),
        "hidden_size": tune.choice([2 ** i for i in range(1, 5)]),
        "dropout_prob": tune.choice(np.arange(start=0., stop=0.5, step=.05).tolist()),
        "bidirectional": tune.choice([True, False]),
        "steps": tune.choice([4 * i for i in range(25, 33)]),
        "bias": False,
        "num_classes": NUM_CLASSES,
        "input_shape": (224, 224),
        "batch_size": 2,
        "lr": tune.loguniform(1e-4, 1e-3),
        "nonlinearity": NonlinearityEnum.SILU,
        "include_additional_transforms": False,
    }

    cur_model_id = ray.put(cur_model)

    result = tune.run(
        partial(
            train_model, 
            epochs=1,
            include_validation=False,
            cnn_model=cur_model_id
        ),
        resources_per_trial={"cpu": os.cpu_count(), "gpu": gpus_per_trial},
        config=lstm_search_space,
        num_samples=5,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("val_acc", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    print(f"Best trial final validation f1: {best_trial.last_result['val_f1']}")
    print(f"Best trial final validation acc: {best_trial.last_result['val_acc']}")

    best_trained_model = CNN_LSTM_Separate(
        cnn_section=cur_model,
        num_lstm_layers=best_trial.config["num_lstm_layers"],
        hidden_size=best_trial.config["hidden_size"],
        num_classes=NUM_CLASSES,
        bidirectional=best_trial.config["bidirectional"],
        steps=best_trial.config["steps"],
        bias=False,
        dropout_prob=best_trial.config["dropout_prob"],
        nonlinearity=best_trial.config["nonlinearity"]
    )

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="val_acc", mode="max")

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    lstm_trial = tune.run(
        partial(
            train_model, 
            epochs=10,
            include_validation=True,
            cnn_model=cur_model_id
        ),
        resources_per_trial={"cpu": os.cpu_count(), "gpu": gpus_per_trial},
        config=result.get_best_config("val_acc", "max", "last"),
        num_samples=1,
        scheduler=scheduler
    )

    best_lstm_checkpoint = lstm_trial.get_best_checkpoint(trial=lstm_trial, metric="val_acc", mode="max")

    with best_lstm_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])


    test_dataset = VideoDataset(
        root=f"{main_folder_path}/data/test", 
        clip_len=best_trial.config["steps"],
        include_additional_transforms=best_trial.config["include_additional_transforms"],
        random_transforms=None
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=int(best_trial.config["batch_size"]),
        num_workers=NUM_WORKERS
    )

    collected_fnames = [os.path.basename(output["path"])[:-4] for output in test_dataset]

    collected_labels, collected_predictions = [], []
    probs = []

    for i, data in enumerate(test_loader):
        vid_inputs, labels = data["video"].to(device), data["target"].to(device)
        labels = labels.type(torch.float32).unsqueeze(dim=1)
        output = best_trained_model(vid_inputs)

        numpy_labels = labels.cpu().numpy().tolist()
        actual_predictions = (output.detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()
        prob = output.detach().cpu().numpy().tolist()

        collected_labels.extend(numpy_labels)
        collected_predictions.extend(actual_predictions)
        probs.extend(prob)

    df = pd.DataFrame({
        "filename": collected_fnames,
        "actual": collected_labels,
        "predicted": collected_predictions,
        "probability": probs
    })

    df.to_csv(f"{main_folder_path}/models/{MODEL_NAME}/results/predictions_{MODEL_NAME}.csv")

    print("Finished entire training regime")

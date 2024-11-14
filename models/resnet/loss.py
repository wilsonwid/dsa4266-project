import matplotlib.pyplot as plt
import pandas as pd

tuning_loss_df = pd.read_csv("models/resnet/results/tuning_loss.csv")
training_loss_df = pd.read_csv("models/resnet/results/training_loss.csv")
tuning_loss_df = tuning_loss_df.rename(
    columns={"epoch": "Epoch", "loss": "Training Loss", "val_loss": "Validation Loss"}
)
training_loss_df["Epoch"] = training_loss_df["Epoch"] + 3

df = pd.concat(
    [tuning_loss_df[["Epoch", "Training Loss", "Validation Loss"]], training_loss_df]
)

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("models/resnet/diagrams/resnet_loss.png", format="png")
plt.show()

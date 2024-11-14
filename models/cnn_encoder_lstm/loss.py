import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("models/cnn_encoder_lstm/results/progress.csv")
df = df.rename(
    columns={
        "iterations_since_restore": "Epoch",
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
    }
)

df["Epoch"] = df["Epoch"] - 1

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN-Encoder-LSTM Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("models/cnn_encoder_lstm/diagrams/cnn_encoder_lstm_loss.png", format="png")
plt.show()

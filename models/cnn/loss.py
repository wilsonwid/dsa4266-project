import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("models/cnn/results/cnn_loss.csv")
df = df.rename(
    columns={
        "epoch": "Epoch",
        "training_loss": "Training Loss",
        "validation_loss": "Validation Loss",
    }
)

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("models/cnn/diagrams/cnn_loss.png", format="png")
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("models/videomae/videomae-loss.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VideoMAE Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("models/videomae/videomae-loss.png", format="png")
plt.show()

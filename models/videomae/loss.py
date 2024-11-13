import matplotlib.pyplot as plt
import pandas as pd

# Note that videomae_loss.csv was manually created from the results automatically pushed to Huggingface:
# https://huggingface.co/shylhy/videomae-large-finetuned-deepfake-subset/blob/2cbeca77c171af587f215a7d262ab6c47cbcea17/README.md

df = pd.read_csv("models/videomae/results/videomae_loss.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VideoMAE Training and Validation Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("models/videomae/videomae_loss.png", format="png")
plt.show()

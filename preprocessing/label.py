import json
import os
import shutil

base_dir = "data"
train_metadata_path = os.path.join(base_dir, "balanced_dataset", "metadata.json")
cropped_dir = os.path.join(base_dir, "cropped")
deepfake_dir = os.path.join(base_dir, "train", "deepfake")
real_dir = os.path.join(base_dir, "train", "real")

os.makedirs(deepfake_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

with open(train_metadata_path, "r") as f:
    metadata = json.load(f)


def copy_files(src_dir, file_suffix=None):
    for filename in os.listdir(src_dir):
        if filename in metadata:
            label = metadata[filename]["label"]
            target_dir = deepfake_dir if label == "FAKE" else real_dir
            new_filename = (
                f"{os.path.splitext(filename)[0]}_{file_suffix}.mp4"
                if file_suffix
                else filename
            )
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(target_dir, new_filename)
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")

if __name__ == "__main__":
    copy_files(cropped_dir)

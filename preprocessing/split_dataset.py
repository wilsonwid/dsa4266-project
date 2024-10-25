import os
import random
import shutil


def move_files(
    source_dir, test_dir, val_dir, test_ratio=0.4, val_ratio=0.3
):  # Ratios are a bit higher than normal to account for augmentation of train dataset
    files = os.listdir(source_dir)
    random.seed(42)
    random.shuffle(files)

    total_files = len(files)
    test_count = int(total_files * test_ratio)
    val_count = int(total_files * val_ratio)

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for filename in files[:test_count]:
        shutil.move(
            os.path.join(source_dir, filename), os.path.join(test_dir, filename)
        )

    for filename in files[test_count : test_count + val_count]:
        shutil.move(os.path.join(source_dir, filename), os.path.join(val_dir, filename))

    print(f"Moved {test_count} files to {test_dir}")
    print(f"Moved {val_count} files to {val_dir}")


base_dir = "data"
categories = ["deepfake", "real"]

for category in categories:
    source_dir = os.path.join(base_dir, "train", category)
    test_dir = os.path.join(base_dir, "test", category)
    val_dir = os.path.join(base_dir, "validation", category)

    move_files(source_dir, test_dir, val_dir)

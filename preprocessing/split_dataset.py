import os
import random
import shutil


def move_files(source_dir, test_dir, val_dir, test_ratio=0.2, val_ratio=0.15):
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


base_dir = "data/train"
categories = ["deepfake", "real"]
test_base_dir = "data/test"
val_base_dir = "data/validation"

for category in categories:
    source_dir = os.path.join(base_dir, category)
    test_dir = os.path.join(test_base_dir, category)
    val_dir = os.path.join(val_base_dir, category)

    move_files(source_dir, test_dir, val_dir)

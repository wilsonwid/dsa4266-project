import os
import random
import shutil

base_dir = "data"
test_dir = os.path.join(base_dir, "test")
validation_dir = os.path.join(base_dir, "validation")

os.makedirs(validation_dir, exist_ok=True)

all_files = os.listdir(test_dir)
num_validation_files = int(len(all_files) * 0.4)

random.seed(42)
validation_files = random.sample(all_files, num_validation_files)

for filename in validation_files:
    src_path = os.path.join(test_dir, filename)
    dest_path = os.path.join(validation_dir, filename)
    shutil.move(src_path, dest_path)
    print(f"Moved {src_path} to {dest_path}")

print(f"{num_validation_files} files moved to the validation folder.")

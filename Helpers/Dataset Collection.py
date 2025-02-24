"""
Dataset Extraction Script

Purpose:
This script randomly selects a specified number of files from the given dataset directories
and copies them into a separate target directory for training and validation.

Functionality:
- It automatically creates the target directories if they don’t exist.
- It ensures random selection of files instead of sequential copying.
- It handles cases where the source directory has fewer files than requested.
- It avoids redundancy by using a single function for different datasets.
"""


import os
import shutil
import random



"""
Change the Parameters accordingly and set paths that suit your system Directories.
"""

# -------------> For Training Data <---------------------
TRAIN_SIZE = 1500
safe_train_source = "Dataset/train/Safe"
notSafe_train_source = "Dataset/train/notSafe"
safe_train_target = "Dataset/Extracted/train/Safe"
notSafe_train_target = "Dataset/Extracted/train/NotSafe"

# -------------> For Validation Data <---------------------
VAL_SIZE = 300
safe_val_source = "Dataset/val1/Safe"
notSafe_val_source = "Dataset/val1/notSafe"
safe_val_target = "Dataset/Extracted/val/Safe"
notSafe_val_target = "Dataset/Extracted/val/NotSafe"


def separate_dataset(source, target, size):
    os.makedirs(target, exist_ok=True) 
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    if len(files) < size:
        print(f"Warning: Source '{source}' has only {len(files)} files. Copying all available.")
        selected_files = files 
    else:
        selected_files = random.sample(files, size) 

    for file in selected_files:
        shutil.copy(os.path.join(source, file), os.path.join(target, file))

    print(f"Copied {len(selected_files)} files from '{source}' to '{target}'")


# Run for both training and validation sets
separate_dataset(safe_train_source, safe_train_target, TRAIN_SIZE)
separate_dataset(notSafe_train_source, notSafe_train_target, TRAIN_SIZE)
separate_dataset(safe_val_source, safe_val_target, VAL_SIZE)
separate_dataset(notSafe_val_source, notSafe_val_target, VAL_SIZE)

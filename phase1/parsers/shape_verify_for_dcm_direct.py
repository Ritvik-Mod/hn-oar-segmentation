import os
import random
import glob
import numpy as np

# Point to the NEW pendrive dataset
dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Master_DIRECT_DCM_new'

# 1. Grab a random .npz file
print("Searching for a random slice...")
# Notice only one '*' before '/data/' because there is no CT1 folder in this batch
all_npz = glob.glob(os.path.join(dataset_path, '*/data/*.npz'))

if not all_npz:
    print("Error: Could not find any .npz files. Check your folder structure!")
else:
    random_file = random.choice(all_npz)

    print(f"\n>>> Inspecting: {random_file} <<<\n")

    # 2. Load the compressed array
    data = np.load(random_file)

    # 3. Print the math
    print("--- Array Contents ---")
    for key in data.files:
        array = data[key]
        print(f"Layer: {key:<15} | Shape: {array.shape} | Type: {array.dtype} | Max Val: {array.max()}")
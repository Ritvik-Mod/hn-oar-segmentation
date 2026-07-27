import os
import random
import glob
import numpy as np

# Point to the master dataset
dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Master'

# 1. Grab a random .npz file
print("Searching for a random slice...")
all_npz = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz'))
random_file = random.choice(all_npz)

print(f"\n>>> Inspecting: {random_file} <<<\n")

# 2. Load the compressed array
data = np.load(random_file)

# 3. Print the math
print("--- Array Contents ---")
for key in data.files:
    array = data[key]
    print(f"Layer: {key:<15} | Shape: {array.shape} | Type: {array.dtype} | Max Val: {array.max()}")
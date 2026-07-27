import os, shutil, json, glob
import numpy as np

source = '/Users/ritvikmod/Desktop/ML_Dataset_Master_DIRECT_DCM_new'
destination = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'
split_path = '/Users/ritvikmod/Desktop/ct scan project/dataset_split.json'
hn_path = '/Users/ritvikmod/Desktop/ct scan project/hn_patients.json'

new_patients = [p for p in os.listdir(source) if p != '.DS_Store']

# Move folders into main dataset
for patient_id in new_patients:
    src = os.path.join(source, patient_id)
    dst = os.path.join(destination, patient_id)
    if os.path.exists(dst):
        print(f"SKIP — already exists: {patient_id}")
        continue
    shutil.copytree(src, dst)
    print(f"Copied: {patient_id}")

# Add all 10 to train split
with open(split_path) as f:
    split = json.load(f)

for patient_id in new_patients:
    if patient_id not in split['train']:
        split['train'].append(patient_id)

with open(split_path, 'w') as f:
    json.dump(split, f, indent=2)

# Update hn_patients.json
with open(hn_path) as f:
    hn_patients = json.load(f)

for patient_id in new_patients:
    if patient_id not in hn_patients:
        hn_patients.append(patient_id)

hn_patients = sorted(hn_patients)
with open(hn_path, 'w') as f:
    json.dump(hn_patients, f, indent=2)

print(f"\nDone.")
print(f"Moved {len(new_patients)} patients to main dataset")
print(f"Train split now: {len(split['train'])} patients")
print(f"Val split: {len(split['val'])} patients")
print(f"Test split: {len(split['test'])} patients")
print(f"Total H&N: {len(hn_patients)} patients")
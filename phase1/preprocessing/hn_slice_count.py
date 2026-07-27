import os, glob, json

dataset_path = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'

with open('/Users/ritvikmod/Desktop/ct scan project/hn_patients.json') as f:
    hn_patients = json.load(f)

total_slices = 0
for patient_id in hn_patients:
    files = glob.glob(f"{dataset_path}/{patient_id}/*/data/*.npz") + \
            glob.glob(f"{dataset_path}/{patient_id}/data/*.npz")
    total_slices += len(files)

print(f"H&N patients: {len(hn_patients)}")
print(f"Total slices for H&N patients: {total_slices}")


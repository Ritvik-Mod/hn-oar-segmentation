import os, glob
import numpy as np

dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Master_DIRECT_DCM_new'

patients = os.listdir(dataset_path)
has_parotid = []
no_parotid = []

for patient_id in patients:
    files = glob.glob(f"{dataset_path}/{patient_id}/data/*.npz")
    found = False
    for f in files[:20]:
        try:
            data = np.load(f)
            if 'PAROTID_R' in data.files or 'PAROTID_L' in data.files:
                found = True
                break
        except: pass
    if found:
        has_parotid.append(patient_id)
    else:
        no_parotid.append(patient_id)

print(f"Have parotid data: {len(has_parotid)}")
print(f"No parotid data:   {len(no_parotid)}")
print(f"\nPatients with parotid: {has_parotid}")
print(f"Patients without:      {no_parotid}")
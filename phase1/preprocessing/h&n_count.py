import os, glob, json
import numpy as np
from collections import defaultdict

dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Master_DIRECT_DCM_new'

HN_ORGANS = {
    'SPINAL_CORD', 'PAROTID_R', 'PAROTID_L', 'BRAINSTEM',
    'EYE_R', 'EYE_L', 'OPTIC_NERVE_R', 'OPTIC_NERVE_L',
    'TEMPORAL_LOBE', 'LARYNX', 'THYROID', 'LENS_R', 'LENS_L',
    'SPINAL_CORD_PRV', 'BRAINSTEM_PRV'
}

# Get all patient IDs and their folder structure
# wc_patients = {os.path.basename(p): p for p in glob.glob(os.path.join(dataset_path, '1~*')) + glob.glob(os.path.join(dataset_path, '[0-9]*'))}

# patient_organ_map = {}

all_npz = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz')) + \
          glob.glob(os.path.join(dataset_path, '*/data/*.npz'))

# Group files by patient
patient_files = defaultdict(list)
for f in all_npz:
    parts = f.replace(dataset_path, '').strip('/').split('/')
    patient_id = parts[0]
    patient_files[patient_id].append(f)

print(f"Total patients: {len(patient_files)}")

hn_patients = []
non_hn_patients = []

for patient_id, files in patient_files.items():
    organs_found = set()
    for f in files[:20]:  # sample first 20 slices per patient for speed
        try:
            data = np.load(f)
            for key in data.files:
                if key != 'image':
                    organs_found.add(key)
        except: pass
    
    has_hn = bool(organs_found & HN_ORGANS)
    if has_hn:
        hn_patients.append(patient_id)
    else:
        non_hn_patients.append({'patient': patient_id, 'organs': list(organs_found)})

print(f"H&N patients:     {len(hn_patients)}")
print(f"Non-H&N patients: {len(non_hn_patients)}")
print(f"\nNon-H&N organ examples:")
for p in non_hn_patients[:10]:
    print(f"  {p['patient']}: {p['organs']}")

# Save both lists
with open('/Users/ritvikmod/Desktop/ct scan project/hn_patients_new10.json', 'w') as f:
    json.dump(sorted(hn_patients), f, indent=2)
with open('/Users/ritvikmod/Desktop/ct scan project/non_hn_patients_new10.json', 'w') as f:
    json.dump(non_hn_patients, f, indent=2)

print(f"\nSaved hn_patients.json and non_hn_patients.json to Desktop")
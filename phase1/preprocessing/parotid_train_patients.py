import json, glob, numpy as np

dataset_path = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'
split_path = '/Users/ritvikmod/Desktop/ct scan project/dataset_split.json'

with open(split_path) as f:
    split = json.load(f)

parotid_train_patients = set()
total = len(split['train'])

for i, patient_id in enumerate(split['train']):
    if i % 50 == 0:
        print(f"  {i}/{total} — found {len(parotid_train_patients)} parotid patients so far")
    
    files = glob.glob(f"{dataset_path}/{patient_id}/*/data/*.npz") + \
            glob.glob(f"{dataset_path}/{patient_id}/data/*.npz")
    
    for f in files:  # check all files this time
        try:
            data = np.load(f)
            if 'PAROTID_R' in data.files or 'PAROTID_L' in data.files:
                parotid_train_patients.add(patient_id)
                break
        except:
            pass

print(f"\nFinal parotid train patients: {len(parotid_train_patients)}")

with open('/Users/ritvikmod/Desktop/ct scan project/parotid_train_patients.json', 'w') as f:
    json.dump(list(parotid_train_patients), f)

print("Saved.")
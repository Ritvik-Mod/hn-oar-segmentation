import json, glob
import numpy as np

dataset_path = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'

with open('/Users/ritvikmod/Desktop/ct scan project/dataset_split.json') as f:
    split = json.load(f)

def has_parotid(patient_id):
    files = glob.glob(f"{dataset_path}/{patient_id}/*/data/*.npz") + \
            glob.glob(f"{dataset_path}/{patient_id}/data/*.npz")
    for f in files[:10]:
        try:
            data = np.load(f)
            if 'PAROTID_R' in data.files or 'PAROTID_L' in data.files:
                return True
        except: pass
    return False

for split_name in ['train', 'val', 'test']:
    count = sum(1 for p in split[split_name] if has_parotid(p))
    print(f"{split_name}: {count}/{len(split[split_name])} patients have parotid annotations")
# import os
# import glob
# import json
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm

# dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Final'

# LABEL_MAP = {
#     'RT_PAROTID': 'PAROTID_R', 'RIGHT_PAROTID': 'PAROTID_R', 'RT PAROTID': 'PAROTID_R',
#     'LEFT_PAROTID': 'PAROTID_L', 'LT_PAROTID': 'PAROTID_L', 'LT PAROTID': 'PAROTID_L',
#     'RT_EYE': 'EYE_R', 'RIGHT_EYE': 'EYE_R',
#     'LT_EYE': 'EYE_L', 'LEFT_EYE': 'EYE_L',
#     'RT_OPTIC_NERVE': 'OPTIC_NERVE_R', 'LT_OPTIC_NERVE': 'OPTIC_NERVE_L',
#     'RT OPTIC NERVE': 'OPTIC_NERVE_R', 'LT OPTIC NERVE': 'OPTIC_NERVE_L',
#     'SPINALCORD': 'SPINAL_CORD', 'CORD': 'SPINAL_CORD',
#     'BRAIN_STEM': 'BRAINSTEM', 'BRAIN STEM': 'BRAINSTEM',
#     'patient': 'BODY', 'PATIENT': 'BODY',
#     'prvspine': 'SPINAL_CORD_PRV', 'PRVSPINE': 'SPINAL_CORD_PRV', 'SPINE_PRV': 'SPINAL_CORD_PRV',
#     'PRVBRAINSTEM': 'BRAINSTEM_PRV',
#     'RT_LENS': 'LENS_R', 'LT_LENS': 'LENS_L',
# }

# DROP_LABELS = {
#     'Foam_Core', 'Carbon_Fiber', 'General', 'sampleElekta', 'HEAD_AND_NECK',
#     '1', 'XX1', 'XX3', '50', 'DARS', 'DX11', 'PTVMAR1', 'Foam Core',
#     'Carbon Fiber', 'BolusFourPoints', 'X1', 'x11', 'P1'
# }

# def process_file(f):
#     try:
#         data = np.load(f)
#         arrays = dict(data)
#         new_arrays = {}
#         changed = False
#         log = []
#         for key, arr in arrays.items():
#             if key == 'image':
#                 new_arrays['image'] = arr
#                 continue
#             if key in DROP_LABELS:
#                 changed = True
#                 log.append(f"DROPPED:{key}")
#                 continue
#             new_key = LABEL_MAP.get(key, key)
#             if new_key != key:
#                 changed = True
#                 log.append(f"RENAMED:{key}->{new_key}")
#             if new_key in new_arrays:
#                 new_arrays[new_key] = np.logical_or(new_arrays[new_key], arr).astype(np.uint8)
#             else:
#                 new_arrays[new_key] = arr
#         if changed:
#             np.savez_compressed(f, **new_arrays)
#             return ('modified', f, log)
#         return ('unchanged', f, [])
#     except Exception as e:
#         return ('error', f, [str(e)])

# if __name__ == '__main__':
#     all_files = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz'))
#     total = len(all_files)
#     modified = skipped = unchanged = 0
#     change_log = {}

#     print(f"Processing {total} files across 8 cores...\n")

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         futures = {executor.submit(process_file, f): f for f in all_files}
#         with tqdm(total=total, unit="file", ncols=80) as pbar:
#             for future in as_completed(futures):
#                 result, filepath, log = future.result()
#                 if result == 'modified':
#                     modified += 1
#                     change_log[filepath] = log
#                 elif result == 'unchanged':
#                     unchanged += 1
#                 else:
#                     skipped += 1
#                 pbar.update(1)

#     # Save the change log so you can audit or reverse if needed
#     log_path = '/Users/ritvikmod/Desktop/label_standardisation_log.json'
#     with open(log_path, 'w') as f:
#         json.dump(change_log, f, indent=2)

#     print(f"\nDone.")
#     print(f"  Modified : {modified}")
#     print(f"  Unchanged: {unchanged}")
#     print(f"  Errors   : {skipped}")
#     print(f"  Change log saved to: {log_path}")

#check

# import os, glob, random
# import numpy as np

# dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Final'
# all_files = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz'))

# # Check 10 random files for any old labels still present
# OLD_LABELS = {'RT_PAROTID', 'LEFT_PAROTID', 'LT_EYE', 'RT_EYE', 'PRVSPINE', 
#               'prvspine', 'BRAIN_STEM', 'Foam_Core', 'Carbon_Fiber', 'HEAD_AND_NECK',
#               'General', 'sampleElekta', 'X1', 'x11', 'P1', '1', 'DARS'}

# sample = random.sample(all_files, 500)
# found_old = {}

# for f in sample:
#     data = np.load(f)
#     for key in data.files:
#         if key in OLD_LABELS:
#             found_old[key] = found_old.get(key, 0) + 1

# if found_old:
#     print("WARNING — old labels still found:")
#     for k, v in found_old.items():
#         print(f"  {k}: {v} files")
# else:
#     print("All clean. No old labels found in sample.")

#running on dicom rt files:

import os, glob, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Master_DIRECT_DCM_new'

LABEL_MAP = {
    'RT_PAROTID': 'PAROTID_R', 'RIGHT_PAROTID': 'PAROTID_R', 'RT PAROTID': 'PAROTID_R',
    'LEFT_PAROTID': 'PAROTID_L', 'LT_PAROTID': 'PAROTID_L', 'LT PAROTID': 'PAROTID_L',
    'RT_EYE': 'EYE_R', 'RIGHT_EYE': 'EYE_R',
    'LT_EYE': 'EYE_L', 'LEFT_EYE': 'EYE_L',
    'RT_OPTIC_NERVE': 'OPTIC_NERVE_R', 'LT_OPTIC_NERVE': 'OPTIC_NERVE_L',
    'RT OPTIC NERVE': 'OPTIC_NERVE_R', 'LT OPTIC NERVE': 'OPTIC_NERVE_L',
    'SPINALCORD': 'SPINAL_CORD', 'CORD': 'SPINAL_CORD',
    'BRAIN_STEM': 'BRAINSTEM', 'BRAIN STEM': 'BRAINSTEM',
    'patient': 'BODY', 'PATIENT': 'BODY',
    'prvspine': 'SPINAL_CORD_PRV', 'PRVSPINE': 'SPINAL_CORD_PRV', 'SPINE_PRV': 'SPINAL_CORD_PRV',
    'PRVBRAINSTEM': 'BRAINSTEM_PRV',
    'RT_LENS': 'LENS_R', 'LT_LENS': 'LENS_L',
}

DROP_LABELS = {
    'Foam_Core', 'Carbon_Fiber', 'General', 'sampleElekta', 'HEAD_AND_NECK',
    '1', 'XX1', 'XX3', '50', 'DARS', 'DX11', 'PTVMAR1', 'Foam Core',
    'Carbon Fiber', 'BolusFourPoints', 'X1', 'x11', 'P1'
}

def process_file(f):
    try:
        data = np.load(f)
        arrays = dict(data)
        new_arrays = {}
        changed = False
        log = []
        for key, arr in arrays.items():
            if key == 'image':
                new_arrays['image'] = arr
                continue
            if key in DROP_LABELS:
                changed = True
                log.append(f"DROPPED:{key}")
                continue
            new_key = LABEL_MAP.get(key, key)
            if new_key != key:
                changed = True
                log.append(f"RENAMED:{key}->{new_key}")
            if new_key in new_arrays:
                new_arrays[new_key] = np.logical_or(new_arrays[new_key], arr).astype(np.uint8)
            else:
                new_arrays[new_key] = arr
        if changed:
            np.savez_compressed(f, **new_arrays)
            return ('modified', f, log)
        return ('unchanged', f, [])
    except Exception as e:
        return ('error', f, [str(e)])

if __name__ == '__main__':
    all_files = glob.glob(os.path.join(dataset_path, '*/data/*.npz'))
    total = len(all_files)
    modified = skipped = unchanged = 0
    change_log = {}

    print(f"Processing {total} DICOM-RT files across 8 cores...\n")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        with tqdm(total=total, unit="file", ncols=80) as pbar:
            for future in as_completed(futures):
                result, filepath, log = future.result()
                if result == 'modified':
                    modified += 1
                    change_log[filepath] = log
                elif result == 'unchanged':
                    unchanged += 1
                else:
                    skipped += 1
                pbar.update(1)

    log_path = '/Users/ritvikmod/Desktop/ct scan project/label_standardisation_log_new10.json'
    with open(log_path, 'w') as f:
        json.dump(change_log, f, indent=2)

    print(f"\nDone.")
    print(f"  Modified : {modified}")
    print(f"  Unchanged: {unchanged}")
    print(f"  Errors   : {skipped}")
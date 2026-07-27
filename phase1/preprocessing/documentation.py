# import os
# import glob
# import pydicom
# import re
# from collections import Counter
# import concurrent.futures
# from tqdm import tqdm

# # Point this to your raw external drive
# raw_data_dir = '/Volumes/Ritvik Mod/AARUNI'
# limit_patients = 200  

# def extract_z_from_name(filename):
#     clean_name = filename.upper().replace('T.', '')
#     match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", clean_name)
#     return float(match.group()) if match else None

# def process_patient(pat_dir_path):
#     """Runs on a single CPU core for one patient."""
#     result = {
#         'spacing': None,
#         'z_range': None,
#         'wc_count': None,
#         'organs': []
#     }
    
#     ct1_path = os.path.join(pat_dir_path, '1~CT1')
#     if not os.path.exists(ct1_path):
#         return result

#     # --- Q3: Aggressive Pixel Spacing Hunt ---
#     for root, _, files in os.walk(pat_dir_path):
#         if result['spacing']: break
#         for f in files:
#             if f.upper().endswith('.DCM') or f.upper().endswith('.CT'):
#                 try:
#                     ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True, force=True)
#                     if hasattr(ds, 'PixelSpacing'):
#                         result['spacing'] = f"{float(ds.PixelSpacing[0]):.3f} x {float(ds.PixelSpacing[1]):.3f}"
#                         break
#                 except: pass

#     # --- Q5 & Q6: Z-Range and WC Counts ---
#     wc_files = glob.glob(os.path.join(ct1_path, '*.[wW][cC]'))
#     if wc_files:
#         result['wc_count'] = len(wc_files)
        
#         z_coords = []
#         for wc in wc_files:
#             z = extract_z_from_name(os.path.basename(wc))
#             if z is not None: z_coords.append(z)
        
#         if len(z_coords) > 1:
#             result['z_range'] = abs(max(z_coords) - min(z_coords))

#     # --- Q7: Top Organs ---
#     contournames_path = os.path.join(ct1_path, 'contournames')
#     if os.path.exists(contournames_path):
#         try:
#             with open(contournames_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 lines = f.read().splitlines()
#             i = 0
#             while i < len(lines):
#                 name = lines[i].strip()
#                 if name and not name.startswith('#'):
#                     if i+1 < len(lines) and ',' in lines[i+1]:
#                         clean_organ_name = name.replace(' ', '_').replace('/', '_').upper()
#                         result['organs'].append(clean_organ_name)
#                 i += 1
#         except: pass

#     return result

# if __name__ == '__main__':
#     print("Finding all patient folders...")
#     patient_folders = []
#     for month_dir in os.scandir(raw_data_dir):
#         if not month_dir.is_dir() or month_dir.name.startswith('.'): continue
#         for pat_dir in os.scandir(month_dir.path):
#             if not pat_dir.is_dir() or pat_dir.name.startswith('.'): continue
#             patient_folders.append(pat_dir.path)
    
#     patient_folders.sort()
    
#     if limit_patients:
#         target_folders = patient_folders[:limit_patients]
#     else:
#         target_folders = patient_folders

#     print(f"Locked in {len(target_folders)} patients. Firing up the M3 performance cores!")

#     all_spacings = []
#     all_z_ranges = []
#     all_wc_counts = []
#     global_organ_counter = Counter()

#     # Multiprocessing with LIVE Progress Bar
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         # Submit all tasks to the CPU cores
#         futures = [executor.submit(process_patient, folder) for folder in target_folders]
        
#         # tqdm creates the visual loading bar. as_completed updates it instantly.
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_folders), desc="Processing Patients", unit="patient"):
#             res = future.result()
#             if res['spacing']: all_spacings.append(res['spacing'])
#             if res['z_range']: all_z_ranges.append(res['z_range'])
#             if res['wc_count']: all_wc_counts.append(res['wc_count'])
#             for organ in res['organs']:
#                 global_organ_counter[organ] += 1

#     print("\n" + "="*40)
#     print("      FINAL DATASET STATISTICS")
#     print("="*40)

#     if all_spacings:
#         print(f"\n[Q3] TOP PIXEL SPACINGS FOUND (Y x X in mm):")
#         for spc, count in Counter(all_spacings).most_common(5):
#             print(f"  - {spc} (found in {count} patients)")
#     else:
#         print("\n[Q3] PIXEL SPACINGS: Still completely missing.")

#     if all_z_ranges:
#         avg_range = sum(all_z_ranges) / len(all_z_ranges)
#         print(f"\n[Q5] Z-COORDINATE RANGE (Anatomical Coverage):")
#         print(f"  - Average Z-Range: {avg_range:.2f} mm")
#         print(f"  - Max Z-Range seen: {max(all_z_ranges):.2f} mm")

#     if all_wc_counts:
#         avg_wc = sum(all_wc_counts) / len(all_wc_counts)
#         print(f"\n[Q6] SLICES / .WC FILES PER PATIENT:")
#         print(f"  - Average .WC files: {avg_wc:.0f} slices")
#         print(f"  - Max .WC files: {max(all_wc_counts)} slices")

#     if global_organ_counter:
#         print(f"\n[Q7] TOP 10 MOST FREQUENT ORGANS (ROIs):")
#         for organ, count in global_organ_counter.most_common(10):
#             print(f"  - {organ} (in {count} patients)")

import os, glob, random
import numpy as np
from collections import Counter

dataset_path = '/Users/ritvikmod/Desktop/ML_Dataset_Final'
all_files = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz'))

# Sample 10,000 random files instead of all 127k
sample = random.sample(all_files, min(10000, len(all_files)))

organ_counter = Counter()
for npz_file in sample:
    try:
        with np.load(npz_file) as data:
            for key in data.files:
                if key != 'image':
                    organ_counter[key] += 1
    except: pass

print(f"Sampled {len(sample)} files\n")
for organ, count in organ_counter.most_common(20):
    print(f"{organ}: {count}")
import os
import glob
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom.uid import ImplicitVRLittleEndian
import re
import concurrent.futures

# --- CONFIG ---
parent_data_dir = '/Volumes/Ritvik Mod/AARUNI'
desktop_dir = '/Users/ritvikmod/Desktop'
master_ml_output = os.path.join(desktop_dir, 'ML_Dataset_Master')
os.makedirs(master_ml_output, exist_ok=True)

# --- HELPERS ---
def extract_z_from_name(filename):
    clean_name = filename.upper().replace('T.', '') 
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", clean_name)
    return float(match.group()) if match else None

def get_global_metadata(pat_dir):
    """Scans a specific patient directory for ANY valid DICOM to borrow spatial metadata."""
    for root, _, files in os.walk(pat_dir):
        for fname in files:
            if fname.upper().endswith('.DCM') or fname.upper().endswith('.CT'):
                try:
                    ds = pydicom.dcmread(os.path.join(root, fname), force=True, stop_before_pixels=True)
                    if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'PixelSpacing'):
                        return {
                            'origin_x': float(ds.ImagePositionPatient[0]),
                            'origin_y': float(ds.ImagePositionPatient[1]),
                            'spacing_y': float(ds.PixelSpacing[0]),
                            'spacing_x': float(ds.PixelSpacing[1])
                        }
                except: pass
    return None

def read_contournames(path):
    organs = {}
    if not os.path.exists(path): return organs
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if name and not name.startswith('#'):
            if i+1 < len(lines) and ',' in lines[i+1]:
                parts = lines[i+1].split(',')
                try: organs[int(parts[0])] = name
                except: pass
        i += 1
    return organs

def parse_wc_file(filepath):
    z = extract_z_from_name(os.path.basename(filepath))
    if z is None: return {}
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    contours = {}
    i = 7  
    while i < len(lines):
        line = lines[i].strip()
        if not (line and '.' not in line): 
            i += 1
            continue
        try:
            n_points = int(line)
            organ_id = int(lines[i+1].strip())
            i += 2
            coords = []
            while i < len(lines) and '.' in lines[i]:
                vals = [float(v) for v in lines[i].replace(',', ' ').split() if v]
                coords.extend(vals)
                i += 1
            if len(coords) >= 4:
                if organ_id not in contours: contours[organ_id] = []
                xy_pairs = [(coords[j], coords[j+1]) for j in range(0, len(coords)-1, 2)]
                contours[organ_id].append({'z': z, 'points': xy_pairs})
        except: i += 1
    return contours

# --- BATCH EXECUTION ---
patient_folders = []

# Iterate through the Month folders first (MAY2024, MAY2025, SEP2024)
for month_dir in os.scandir(parent_data_dir):
    if month_dir.is_dir() and not month_dir.name.startswith('.'):
        # Now collect the actual patient folders inside them
        for pat_dir in os.scandir(month_dir.path):
            if pat_dir.is_dir() and not pat_dir.name.startswith('.'):
                patient_folders.append(pat_dir.path)

# Sort them alphabetically/numerically
patient_folders.sort()
print(f"Found {len(patient_folders)} patients across all month folders in {parent_data_dir}")

for p_path in patient_folders:
    p_id = os.path.basename(p_path)
    
    # 1. Strictly target 'CT1' and ignore 'CT2' or other variations
    ct_folders = [f.path for f in os.scandir(p_path) if f.is_dir() and 'CT1' in f.name.upper() and 'CT2' not in f.name.upper()]

    # If no valid CT1 folder exists, skip the patient silently
    if not ct_folders:
        continue

    print(f"\n>>> PROCESSING PATIENT: {p_id} <<<")
    global_meta = get_global_metadata(p_path)

    for ct_path in ct_folders:
        ct_name = os.path.basename(ct_path)
        
        all_wc = sorted(glob.glob(os.path.join(ct_path, 'T.*.WC')))
        
        dcm_dict = {}
        for root, _, files in os.walk(ct_path):
            for f in files:
                if (f.upper().endswith('.DCM') or f.upper().endswith('.CT')) and not f.upper().endswith('.WC'):
                    z = extract_z_from_name(f)
                    if z is not None: dcm_dict[z] = os.path.join(root, f)

        # 2. Pre-flight check: Do we have both files AND matching Z-coordinates?
        if not all_wc or not dcm_dict:
            print(f"  Skipping: {ct_name} (Missing required .WC or DICOM files)")
            continue
            
        has_match = any(extract_z_from_name(os.path.basename(wc)) in dcm_dict for wc in all_wc)
        if not has_match:
            print(f"  Skipping: {ct_name} (No matching Z-coordinates found between images and contours)")
            continue

        # Paths are defined, but NOT created yet
        ct_output_root = os.path.join(master_ml_output, p_id, ct_name)
        data_dir = os.path.join(ct_output_root, 'data')
        verify_dir = os.path.join(ct_output_root, 'verification')
        
        folders_created = False
        organs = read_contournames(os.path.join(ct_path, 'contournames'))

        for wc_file in all_wc:
            z_c = extract_z_from_name(os.path.basename(wc_file))
            if z_c is None or not dcm_dict: continue
            
            # --- FUZZY MATCH INJECTION ---
            closest_z = min(dcm_dict.keys(), key=lambda k: abs(k - z_c))
            if abs(closest_z - z_c) > 0.5: continue
            # -----------------------------
            
            try:
                # Use closest_z instead of z_c to read the file
                ds = pydicom.dcmread(dcm_dict[closest_z], force=True)
                if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                
                try: image_array = ds.pixel_array
                except:
                    rows, cols = getattr(ds, 'Rows', 512), getattr(ds, 'Columns', 512)
                    with open(dcm_dict[closest_z], 'rb') as f:
                        f.seek(os.path.getsize(dcm_dict[closest_z]) - (rows*cols*2))
                        image_array = np.frombuffer(f.read(), dtype=np.int16).reshape(rows, cols)

                ox = float(ds.ImagePositionPatient[0]) if hasattr(ds, 'ImagePositionPatient') else global_meta['origin_x']
                oy = float(ds.ImagePositionPatient[1]) if hasattr(ds, 'ImagePositionPatient') else global_meta['origin_y']
                sy = float(ds.PixelSpacing[0]) if hasattr(ds, 'PixelSpacing') else global_meta['spacing_y']
                sx = float(ds.PixelSpacing[1]) if hasattr(ds, 'PixelSpacing') else global_meta['spacing_x']

                slice_masks = {}
                contours = parse_wc_file(wc_file)
                for o_id, c_list in contours.items():
                    if o_id in [27, 28, 29]: continue
                    mask = np.zeros(image_array.shape, dtype=np.uint8)
                    for c in c_list:
                        pts = np.array(c['points'])
                        px, py = (pts[:, 0] - ox) / sx, (-pts[:, 1] - oy) / sy
                        cv2.fillPoly(mask, [np.vstack((px, py)).T.astype(np.int32)], 1)
                    name = organs.get(o_id, f"ID_{o_id}").replace(' ', '_').replace('/', '_')
                    slice_masks[name] = mask

                if slice_masks:
                    # 3. Deferred Folder Creation: Only make directories right before a valid save
                    if not folders_created:
                        os.makedirs(data_dir, exist_ok=True)
                        os.makedirs(verify_dir, exist_ok=True)
                        folders_created = True

                    fname = f"Z_{z_c}"
                    np.savez_compressed(os.path.join(data_dir, f"{fname}.npz"), image=image_array, **slice_masks)
                    
                    # Verification Plot with Legends
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_array, cmap='gray')
                    cmap = plt.get_cmap('hsv', len(slice_masks) + 1)
                    
                    handles = []
                    for i, (name, mask) in enumerate(slice_masks.items()):
                        if np.any(mask):
                            color = cmap(i)
                            plt.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)
                            line, = plt.plot([], [], color=color, label=name)
                            handles.append(line)
                    
                    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
                    plt.title(f"{p_id} - {ct_name} - {fname}")
                    plt.axis('off')
                    plt.savefig(os.path.join(verify_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
                    plt.close()
            except: pass

print("\n--- BATCH COMPLETE ---")
import os
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom.uid import ImplicitVRLittleEndian
import concurrent.futures

# --- CONFIG ---
parent_data_dir = '/Users/ritvikmod/Desktop/new 10 patient data'
desktop_dir = '/Users/ritvikmod/Desktop'
master_ml_output = os.path.join(desktop_dir, 'ML_Dataset_Master_DIRECT_DCM_new')
os.makedirs(master_ml_output, exist_ok=True)

# --- PATIENT PROCESSING FUNCTION ---
def process_patient(p_path):
    p_id = os.path.basename(p_path)
    
    # --- 1. RESUME LOGIC ---
    patient_output_dir = os.path.join(master_ml_output, p_id)
    if os.path.exists(patient_output_dir):
        return f"Already Done: {p_id}"
        
    data_dir = os.path.join(patient_output_dir, 'data')
    verify_dir = os.path.join(patient_output_dir, 'verification')

    ct_slices = []
    rt_file = None

    # --- 2. FAST-SCAN DIRECTORY ---
    # We only store the file PATHS here to prevent RAM overload during multiprocessing
    for f in os.listdir(p_path):
        if f.startswith('.'): continue
        filepath = os.path.join(p_path, f)
        if not os.path.isfile(filepath): continue
        
        try:
            # stop_before_pixels=True makes scanning 100x faster
            ds = pydicom.dcmread(filepath, force=True, stop_before_pixels=True)
            modality = getattr(ds, 'Modality', '')
            
            if modality == 'RTSTRUCT':
                # Re-read fully only for the RTSTRUCT file to grab the heavy sequence data
                rt_file = pydicom.dcmread(filepath, force=True)
            elif modality == 'CT' and hasattr(ds, 'ImagePositionPatient'):
                ct_slices.append(filepath)
        except: pass

    if not rt_file:
        return f"Skipped: {p_id} (No RTSTRUCT file found)"
    if not ct_slices:
        return f"Skipped: {p_id} (No CT files found)"

    # --- 3. PARSE 3D CONTOURS ---
    roi_name_map = {roi.ROINumber: roi.ROIName for roi in getattr(rt_file, 'StructureSetROISequence', [])}
    contours_by_organ = {}
    
    for roi_contour in getattr(rt_file, 'ROIContourSequence', []):
        roi_num = roi_contour.ReferencedROINumber
        organ_name = roi_name_map.get(roi_num, f"ROI_{roi_num}")
        contours_by_organ[organ_name] = []
        
        for contour in getattr(roi_contour, 'ContourSequence', []):
            points = np.array(contour.ContourData).reshape((-1, 3))
            contours_by_organ[organ_name].append({
                'z': points[0, 2],
                'points': points[:, :2]
            })

    # --- 4. EXTRACT MASKS & VERIFICATION ---
    success_count = 0
    folders_created = False
    plt.switch_backend('Agg') # Thread safety for plots
    
    for ct_path in ct_slices:
        try:
            ds = pydicom.dcmread(ct_path, force=True)
            z_slice = float(ds.ImagePositionPatient[2])
            
            if not hasattr(ds, 'file_meta'):
                ds.file_meta = pydicom.dataset.FileMetaDataset()
            if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                
            try:
                image_array = ds.pixel_array
            except Exception:
                rows, cols = getattr(ds, 'Rows', 512), getattr(ds, 'Columns', 512)
                with open(ct_path, 'rb') as f:
                    f.seek(os.path.getsize(ct_path) - (rows*cols*2))
                    image_array = np.frombuffer(f.read(), dtype=np.int16).reshape(rows, cols)
            
            ox, oy = float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1])
            sy, sx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
            
            slice_masks = {}
            has_content = False
            
            for organ_name, contour_list in contours_by_organ.items():
                mask = np.zeros(image_array.shape, dtype=np.uint8)
                drawn = False
                
                for contour in contour_list:
                    if abs(contour['z'] - z_slice) < 0.5:
                        pts = contour['points']
                        px = (pts[:, 0] - ox) / sx
                        py = (pts[:, 1] - oy) / sy
                        poly = np.vstack((px, py)).T.astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                        drawn = True
                        
                if drawn:
                    safe_name = organ_name.replace(' ', '_').replace('/', '_')
                    slice_masks[safe_name] = mask
                    has_content = True
                    
            if has_content:
                if not folders_created:
                    os.makedirs(data_dir, exist_ok=True)
                    os.makedirs(verify_dir, exist_ok=True)
                    folders_created = True

                fname = f"Z_{z_slice:.1f}"
                
                # Bundle image and masks together for the .npz file
                save_dict = {'image': image_array}
                save_dict.update(slice_masks)
                np.savez_compressed(os.path.join(data_dir, f"{fname}.npz"), **save_dict)
                
                # Plotting logic
                plt.figure(figsize=(10, 10))
                plt.imshow(image_array, cmap='gray')
                cmap = plt.get_cmap('hsv', len(slice_masks))
                
                handles = []
                color_idx = 0
                for name, mask in slice_masks.items():
                    color = cmap(color_idx)
                    plt.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)
                    line, = plt.plot([], [], color=color, label=name)
                    handles.append(line)
                    color_idx += 1
                
                plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
                plt.title(f"{p_id} - {fname}")
                plt.axis('off')
                plt.savefig(os.path.join(verify_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
                plt.close()
                success_count += 1
                
        except Exception:
            pass

    return f"Finished {p_id}: Extracted {success_count} valid slices."

# --- MULTIPROCESSING TRIGGER ---
if __name__ == '__main__':
    patient_folders = [f.path for f in os.scandir(parent_data_dir) if f.is_dir() and not f.name.startswith('.')]
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patients on the Pendrive.")
    print("Kicking off RTSTRUCT Multiprocessing...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_patient, patient_folders)
        for res in results:
            if res:
                print(res)

    print("\n--- PENDRIVE BATCH COMPLETE ---")
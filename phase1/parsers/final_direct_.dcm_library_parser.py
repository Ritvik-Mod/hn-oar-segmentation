import os
import numpy as np
import cv2
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
import matplotlib.pyplot as plt

# --- CONFIG ---
patient_id = "RAJUBHAI_RATHOD"
dicom_dir = '/Users/ritvikmod/Desktop/RAJUBHAI RATHOD'
desktop_dir = '/Users/ritvikmod/Desktop'
master_ml_output = os.path.join(desktop_dir, 'ML_Dataset_Master')

data_dir = os.path.join(master_ml_output, patient_id, 'CT1', 'data')
verify_dir = os.path.join(master_ml_output, patient_id, 'CT1', 'verification')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(verify_dir, exist_ok=True)

try:
    print(f"\n>>> PROCESSING CUSTOM DICOM-RT: {patient_id} <<<")
    
    ct_slices = []
    rt_file = None

    # 1. Force load files
    for f in os.listdir(dicom_dir):
        if f.startswith('.'): continue
        filepath = os.path.join(dicom_dir, f)
        try:
            ds = pydicom.dcmread(filepath, force=True)
            modality = getattr(ds, 'Modality', '')
            if modality == 'RTSTRUCT':
                rt_file = ds
            elif modality == 'CT' and hasattr(ds, 'ImagePositionPatient'):
                ct_slices.append(ds)
        except:
            pass

    if not rt_file:
        raise ValueError("Could not load RTSTRUCT file.")
    if not ct_slices:
        raise ValueError("Could not load CT files.")

    print(f"Loaded {len(ct_slices)} CT slices and 1 RTSTRUCT file.")

    # 2. Extract Organ Names
    roi_name_map = {roi.ROINumber: roi.ROIName for roi in getattr(rt_file, 'StructureSetROISequence', [])}
    
    # 3. Parse Contours into Memory
    print("Mapping 3D contours...")
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

    # 4. Draw Masks onto CT Slices
    success_count = 0
    print("Extracting masks to NPZ...")
    for ds in ct_slices:
        z_slice = float(ds.ImagePositionPatient[2])
        
        # --- FIX: Inject Transfer Syntax and Fallback ---
        if not hasattr(ds, 'file_meta'):
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            
        try:
            image_array = ds.pixel_array
        except Exception as e:
            # Fallback for completely missing pixel data format
            rows, cols = getattr(ds, 'Rows', 512), getattr(ds, 'Columns', 512)
            filepath = ds.filename
            with open(filepath, 'rb') as f:
                f.seek(os.path.getsize(filepath) - (rows*cols*2))
                image_array = np.frombuffer(f.read(), dtype=np.int16).reshape(rows, cols)
        # ------------------------------------------------
        
        ox, oy = float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1])
        sy, sx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
        
        slice_data = {'image': image_array}
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
                slice_data[safe_name] = mask
                has_content = True
                
        if has_content:
            fname = f"Z_{z_slice:.1f}"
            np.savez_compressed(os.path.join(data_dir, f"{fname}.npz"), **slice_data)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image_array, cmap='gray')
            cmap = plt.get_cmap('hsv', len(slice_data))
            
            handles = []
            color_idx = 0
            for name, mask in slice_data.items():
                if name != 'image':
                    color = cmap(color_idx)
                    plt.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)
                    line, = plt.plot([], [], color=color, label=name)
                    handles.append(line)
                    color_idx += 1
            
            plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
            plt.title(f"{patient_id} - {fname}")
            plt.axis('off')
            plt.savefig(os.path.join(verify_dir, f"{fname}.png"), bbox_inches='tight', dpi=150)
            plt.close()
            success_count += 1

    print(f"--- BATCH COMPLETE: {success_count} slices extracted to {data_dir} ---")

except Exception as e:
    print(f"Error processing data: {e}")
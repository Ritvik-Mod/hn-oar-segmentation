import os
import pydicom

def find_and_read_dicom(base_path):
    print(f"Scanning external drive: '{base_path}' for a DICOM file...")
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Look for standard DICOM files
            if file.upper().endswith('.DCM') or 'image' in file.lower():
                dcm_path = os.path.join(root, file)
                try:
                    # Read the header without loading the heavy pixels
                    dataset = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    
                    # If it has the math tags, print them and stop!
                    if 'RescaleIntercept' in dataset:
                        print(f"\n[SUCCESS] Found Raw DICOM: {dcm_path}")
                        print("\n=== THE HOSPITAL'S EXACT MATH ===")
                        print(f"Scanner Manufacturer : {dataset.get('Manufacturer', 'Unknown')}")
                        print(f"Rescale Intercept    : {dataset.RescaleIntercept}")
                        print(f"Rescale Slope        : {dataset.RescaleSlope}")
                        return
                except Exception:
                    # Not a valid DICOM, keep searching
                    continue
                    
    print("\nCould not find a valid CT DICOM with Rescale tags on the drive.")

if __name__ == '__main__':
    find_and_read_dicom("/Volumes/Ritvik Mod/AARUNI")
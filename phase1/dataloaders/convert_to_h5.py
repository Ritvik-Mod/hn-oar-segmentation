import os
import glob
import numpy as np
import h5py

def main():
    dataset_path = '/home/<hpc-user>/project/data/ML_Dataset_Final'
    output_h5    = '/home/<hpc-user>/project/data/dataset.h5'

    print("Scanning directory for .npz files...")
    # Find all npz files (handling both nested and flat folder structures)
    files = glob.glob(os.path.join(dataset_path, '*/*/data/*.npz')) + \
            glob.glob(os.path.join(dataset_path, '*/data/*.npz'))
    
    print(f"Found {len(files)} files. Starting HDF5 conversion...")

    # Open the HDF5 file in write mode
    with h5py.File(output_h5, 'w') as hf:
        for i, filepath in enumerate(files):
            # Print a heartbeat every 500 files
            if i % 500 == 0:
                print(f"  Packed {i} / {len(files)} files...", flush=True)
            
            # Create a clean key (e.g., "1~240193/1~CT1/data/Z_-1308.3.npz")
            key = filepath.replace(dataset_path, '').strip('/')
            
            # Load the raw numpy arrays
            try:
                with np.load(filepath) as data:
                    # Create a folder inside the HDF5 file for this specific scan
                    grp = hf.create_group(key)
                    
                    # Save the image array (lzf compression is lightning fast)
                    grp.create_dataset('image', data=data['image'], compression='lzf')
                    
                    # Save the masks if they exist
                    if 'PAROTID_R' in data.files:
                        grp.create_dataset('PAROTID_R', data=data['PAROTID_R'], compression='lzf')
                    if 'PAROTID_L' in data.files:
                        grp.create_dataset('PAROTID_L', data=data['PAROTID_L'], compression='lzf')
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    print("\nSUCCESS: Entire dataset packed into dataset.h5!")

if __name__ == '__main__':
    main()

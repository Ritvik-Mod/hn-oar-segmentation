import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
dataset_dir = '/Users/ritvikmod/Desktop/1~200734/1~CT1/ML_Dataset'
verify_dir = '/Users/ritvikmod/Desktop/1~200734/1~CT1/ML_Verification'

os.makedirs(verify_dir, exist_ok=True)

npz_files = sorted(glob.glob(os.path.join(dataset_dir, '*.npz')))
print(f"Found {len(npz_files)} .npz files. Generating verification images...")

count = 0
for npz_path in npz_files:
    # Load the compressed arrays
    data = np.load(npz_path)
    img = data['image']
    
    # Grab all the keys that aren't the base image
    mask_keys = [k for k in data.files if k != 'image']
    
    # Only generate an image if this slice actually has masks (saves time)
    has_organs = any(np.any(data[k]) for k in mask_keys)
    if not has_organs:
        continue
        
    plt.figure(figsize=(10, 10))
    # Plot the raw CT array
    plt.imshow(img, cmap='gray')
    
    # Setup colors for the overlays
    colors = plt.cm.get_cmap('hsv', max(len(mask_keys), 1))
    valid_handles = []
    
    for i, key in enumerate(mask_keys):
        mask = data[key]
        if np.any(mask): 
            # plt.contour draws a crisp line exactly at the boundary of the 0s and 1s
            plt.contour(mask, levels=[0.5], colors=[colors(i)], linewidths=2)
            
            # Create a dummy line for the legend
            line, = plt.plot([], [], color=colors(i), label=key)
            valid_handles.append(line)
    
    if valid_handles:
        plt.legend(handles=valid_handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
        
    filename = os.path.basename(npz_path).replace('.npz', '.png')
    plt.title(f"Dataset Verification: {filename}")
    plt.axis('off')
    
    # Save and close (closing is important so you don't run out of RAM!)
    save_path = os.path.join(verify_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() 
    
    count += 1

print(f"Done! Saved {count} verification images to: {verify_dir}")
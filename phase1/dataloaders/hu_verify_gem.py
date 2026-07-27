import numpy as np
import matplotlib.pyplot as plt
import glob

def verify_correct_math(npz_path):
    print(f"Reading: {npz_path}")
    raw_image = np.load(npz_path)['image']
    
    # 1. Apply the Official DICOM Math
    image_hu = (raw_image.astype(np.float32) * 1.0) - 8192.0
    
    # 2. Apply Soft Tissue Window
    hu_min, hu_max = -150.0, 250.0
    windowed_image = np.clip(image_hu, hu_min, hu_max)
    
    # 3. Normalize to [0.0, 1.0]
    normalized_image = (windowed_image - hu_min) / (hu_max - hu_min)

    print("\n--- NEW ARRAY STATS ---")
    print(f"Raw Mean       : {raw_image.mean():.2f} (Before Math)")
    print(f"True HU Mean   : {image_hu.mean():.2f} (After -8192 Shift)")
    print(f"Final NN Input : Min {normalized_image.min():.2f} to Max {normalized_image.max():.2f}")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title(f"Raw Image\nMax: {raw_image.max():.0f}")
    axes[0].axis('off')
    
    axes[1].imshow(windowed_image, cmap='gray')
    axes[1].set_title(f"Windowed (-150 to 250 HU)\nHU Min: {windowed_image.min():.0f}, Max: {windowed_image.max():.0f}")
    axes[1].axis('off')
    
    axes[2].imshow(normalized_image, cmap='gray')
    axes[2].set_title(f"Neural Net Input\nMin: {normalized_image.min():.1f}, Max: {normalized_image.max():.1f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = 'hu_diagnostic_fixed.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved visual diagnosis to {save_path}")

if __name__ == '__main__':
    test_file = glob.glob('/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final/*/1~CT1/data/*.npz')[0]
    verify_correct_math(test_file)
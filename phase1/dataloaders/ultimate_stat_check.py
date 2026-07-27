import numpy as np
import matplotlib.pyplot as plt
import glob

def ultimate_stat_check(npz_path):
    print(f"Reading: {npz_path}")
    raw_image = np.load(npz_path)['image'].astype(np.float32)

    # 1. Apply the Math
    image_hu = raw_image - 8192.0
    hu_min, hu_max = -150.0, 250.0
    windowed_image = np.clip(image_hu, hu_min, hu_max)
    normalized_image = (windowed_image - hu_min) / (hu_max - hu_min)

    # 2. Deep Statistical Profiling
    total_pixels = normalized_image.size # Should be 512 x 512 = 262,144
    
    # Categorize the pixels
    air_pixels = np.sum(normalized_image == 0.0)
    bone_pixels = np.sum(normalized_image == 1.0)
    tissue_pixels = np.sum((normalized_image > 0.0) & (normalized_image < 1.0))

    print("\n=== PIXEL DISTRIBUTION ANALYSIS ===")
    print(f"Total Pixels in Slice : {total_pixels:,}")
    print(f"Air/Background (0.0)  : {air_pixels:,} ({air_pixels/total_pixels*100:.2f}%)")
    print(f"Bone/Dense (1.0)      : {bone_pixels:,} ({bone_pixels/total_pixels*100:.2f}%)")
    print(f"Soft Tissue (0.0-1.0) : {tissue_pixels:,} ({tissue_pixels/total_pixels*100:.2f}%)")

    # 3. Hard Mathematical Assertions (Will crash if the math is broken)
    assert air_pixels + bone_pixels + tissue_pixels == total_pixels, "FATAL: Pixel counts do not add up!"
    assert normalized_image.min() == 0.0, "FATAL: Array minimum is not exactly 0.0"
    assert normalized_image.max() == 1.0, "FATAL: Array maximum is not exactly 1.0"
    print("\n[PASS] All mathematical assertions passed. Tensors are strictly bounded.")

    # 4. Plot Logarithmic Histograms to see the actual curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(raw_image.flatten(), bins=100, color='gray')
    axes[0].set_title('1. Raw Hospital Pixels')
    axes[0].set_yscale('log') # Log scale because air dominates the image

    axes[1].hist(image_hu.flatten(), bins=100, color='blue')
    axes[1].set_title('2. True HU (-8192 Shift)')
    axes[1].axvline(x=-150, color='r', linestyle='--', label='Window Min (-150)')
    axes[1].axvline(x=250, color='g', linestyle='--', label='Window Max (250)')
    axes[1].legend()
    axes[1].set_yscale('log')

    axes[2].hist(normalized_image.flatten(), bins=50, color='green')
    axes[2].set_title('3. Final PyTorch Tensor [0.0 to 1.0]')
    axes[2].set_yscale('log')

    plt.tight_layout()
    save_path = 'ultimate_stat_check.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved statistical histograms to {save_path}")

if __name__ == '__main__':
    test_file = glob.glob('/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final/*/1~CT1/data/*.npz')[0]
    ultimate_stat_check(test_file)
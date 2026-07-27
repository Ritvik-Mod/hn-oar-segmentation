import numpy as np
import matplotlib.pyplot as plt
import glob

def reverse_engineer_scale():
    # Grab the same test slice
    npz_path = glob.glob('/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final/*/1~CT1/data/*.npz')[0]
    raw_image = np.load(npz_path)['image']
    
    # Filter out the 0s (the black corners outside the scanner FOV)
    # We only want to analyze the actual patient and the air inside the tube
    valid_pixels = raw_image[raw_image > 100].flatten()
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(valid_pixels, bins=250, color='teal')
    plt.title('Pixel Value Distribution (Finding Air and Soft Tissue Peaks)')
    plt.xlabel('Raw Hospital Pixel Value')
    plt.ylabel('Number of Pixels')
    plt.grid(True, alpha=0.3)
    
    save_path = '/Users/ritvikmod/Desktop/ct scan project/hu_histogram.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved histogram to {save_path}\n")
    
    # Automatically find the highest peaks
    print("=== PEAK ANALYSIS ===")
    sorted_bins = sorted(zip(counts, bins), reverse=True)
    
    # The absolute biggest peak will be Air (since most of the tube is empty)
    air_peak = sorted_bins[0][1]
    print(f"1. Largest Peak (Likely Air inside scanner): ~{air_peak:.0f}")
    
    # Let's find the next major peak that is significantly higher than Air (likely soft tissue)
    tissue_peaks = [b for c, b in sorted_bins if b > air_peak + 500]
    if tissue_peaks:
        print(f"2. Secondary Peak (Likely Soft Tissue/Water): ~{tissue_peaks[0]:.0f}")

if __name__ == '__main__':
    reverse_engineer_scale()
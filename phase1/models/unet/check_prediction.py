import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import random

from unet import UNet 

def window_and_normalise(image):
    hu_intercept = -8192.0
    hu_slope = 1.0
    hu_min = -150.0
    hu_max = 250.0
    image_hu = image.astype(np.float32) * hu_slope + hu_intercept
    image_clipped = np.clip(image_hu, hu_min, hu_max)
    return (image_clipped - hu_min) / (hu_max - hu_min)

def main():
    print("Loading model weights...")
    device = torch.device('cpu') 
    model = UNet(in_channels=1, out_channels=2).to(device)
    
    weights_path = '/home/<hpc-user>/project/checkpoints/unet/best_model.pth'
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded weights from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")

    print("Fetching a random validation slice...")
    h5_path = '/home/<hpc-user>/project/data/dataset.h5'
    split_path = '/home/<hpc-user>/project/dataset_split.json'
    
    with open(split_path) as f:
        val_patients = json.load(f)['val']

    with h5py.File(h5_path, 'r') as hf:
        valid_slices = []
        def collect_val(name):
            if name.endswith('.npz') and name.split('/')[0] in val_patients:
                if 'PAROTID_R' in hf[name] or 'PAROTID_L' in hf[name]:
                    valid_slices.append(name)
        hf.visit(collect_val)
        
        chosen_slice = random.choice(valid_slices)
        print(f"Testing on slice: {chosen_slice}")
        
        grp = hf[chosen_slice]
        image_raw = grp['image'][...]
        
        mask_true = np.zeros((2, 512, 512), dtype=np.float32)
        if 'PAROTID_R' in grp: mask_true[0] = grp['PAROTID_R'][...]
        if 'PAROTID_L' in grp: mask_true[1] = grp['PAROTID_L'][...]

    image_norm = window_and_normalise(image_raw)
    image_tensor = torch.tensor(image_norm).unsqueeze(0).unsqueeze(0).to(device) 

    print("Running prediction...")
    with torch.no_grad():
        output = model(image_tensor)
        # Getting raw probabilities between 0.0 and 1.0. No rounding!
        pred_prob = torch.sigmoid(output).squeeze(0).numpy() 

    print("Saving visualization...")
    true_combined = mask_true[0] + mask_true[1]
    pred_combined = pred_prob[0] + pred_prob[1] 

    print(f"--- RAW MATH ---")
    print(f"Max Confidence in this slice: {pred_combined.max():.6f}")
    print(f"Average Confidence: {pred_combined.mean():.6f}")
    print(f"----------------")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_norm, cmap='gray')
    axes[0].set_title("Original CT Slice")
    axes[1].imshow(true_combined, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    
    # Using 'inferno' colormap. 0.0 is black, 0.5 is orange, 1.0 is yellow/white.
    im = axes[2].imshow(pred_combined, cmap='inferno')
    axes[2].set_title("U-Net Confidence Heatmap")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    for ax in axes[:2]: ax.axis('off') # Keep axes off for the first two, but leave it for the colorbar math
    axes[2].axis('off')
    
    save_path = '/home/<hpc-user>/project/prediction_test.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"SUCCESS! Image saved to {save_path}")

if __name__ == '__main__':
    main()

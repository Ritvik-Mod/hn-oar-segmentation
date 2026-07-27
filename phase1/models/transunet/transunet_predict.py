import os
import random
import torch
import matplotlib.pyplot as plt

from hdf5_dataloader import ParotidDataset
from transunet import TransUNet

def main():
    # ── Force CPU to protect the live training job ──
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # ── Paths ──
    dataset_path    = '/home/<hpc-user>/project/data/dataset.h5'
    split_path      = '/home/<hpc-user>/project/dataset_split.json'
    checkpoint_path = '/home/<hpc-user>/project/checkpoints/transunet/best_model.pth'

    # ── Load Model ──
    print("Loading TransUNet...")
    model = TransUNet(in_channels=1, out_channels=2).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded weights from Epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")

    # ── Load Validation Data ──
    print("Loading Validation Dataset...")
    val_ds = ParotidDataset('val', dataset_path, split_path, augment=False)

    # ── Find a RANDOM slice with an actual Parotid Gland ──
    print("Hunting for a random valid slice...")
    sample_idx = -1
    
    # Create a list of all possible indices and shuffle them
    all_indices = list(range(len(val_ds)))
    random.shuffle(all_indices)
    
    for i in all_indices:
        _, mask = val_ds[i]
        if mask.sum() > 100:  # Ensure the gland is visible
            sample_idx = i
            break
            
    if sample_idx == -1:
        print("Could not find a validation slice with a mask.")
        return

    image, mask = val_ds[sample_idx]
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = image.unsqueeze(0).to(device)
    
    # ── Predict ──
    print("Generating TransUNet prediction...")
    with torch.no_grad():
        output = model(img_tensor)
        # Apply Sigmoid for independent channels and threshold at 0.5
        pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy()

    # Convert tensors to numpy arrays for matplotlib
    img_np  = image.squeeze(0).cpu().numpy()
    
    # Combine Left and Right parotid channels for plotting
    mask_np = mask[0].cpu().numpy() + mask[1].cpu().numpy()
    pred_np = pred[0] + pred[1]

    # ── Plotting ──
    print("Saving visualization...")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Raw CT
    plt.subplot(1, 3, 1)
    plt.title("Original CT Scan")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    
    # Plot 2: Ground Truth (Red)
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(img_np, cmap='gray')
    plt.imshow(mask_np, cmap='Reds', alpha=0.5)
    plt.axis('off')
    
    # Plot 3: Model Prediction (Blue)
    plt.subplot(1, 3, 3)
    plt.title(f"Prediction (Epoch {checkpoint['epoch']})")
    plt.imshow(img_np, cmap='gray')
    plt.imshow(pred_np, cmap='Blues', alpha=0.5)
    plt.axis('off')
    
    save_path = 'transunet_epoch_test.png'
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Done! Saved to {save_path}")

if __name__ == '__main__':
    main()

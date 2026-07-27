import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import matplotlib.pyplot as plt


class ParotidDataset(Dataset):
    def __init__(self, split, dataset_path, split_json_path, augment=False):
        self.dataset_path = dataset_path
        self.augment = augment

        # Correct HU conversion: raw pixel * 1.0 - 8192
        self.hu_intercept = -8192.0
        self.hu_slope = 1.0

        # Soft tissue window for H&N
        self.hu_min = -150.0
        self.hu_max = 250.0

        with open(split_json_path) as f:
            split_data = json.load(f)
        self.patients = split_data[split]

        self.slices = []
        for patient_id in self.patients:
            files = glob.glob(os.path.join(dataset_path, patient_id, '*', 'data', '*.npz')) + \
                    glob.glob(os.path.join(dataset_path, patient_id, 'data', '*.npz'))
            self.slices.extend(files)

        print(f"{split}: {len(self.patients)} patients, {len(self.slices)} slices")

    def __len__(self):
        return len(self.slices)

    def window_and_normalise(self, image):
        # Step 1: Convert raw uint16 to HU using correct intercept
        image_hu = image.astype(np.float32) * self.hu_slope + self.hu_intercept
        # Step 2: Clip to soft tissue window
        image_clipped = np.clip(image_hu, self.hu_min, self.hu_max)
        # Step 3: Normalise to [0, 1]
        image_norm = (image_clipped - self.hu_min) / (self.hu_max - self.hu_min)
        return image_norm

    def augment_sample(self, image, mask_r, mask_l):
        # Only horizontal flip is anatomically valid for CT
        # Vertical flip and rotation destroy spatial priors (spine always at bottom)
        do_hflip = random.random() > 0.5

        if do_hflip:
            image = np.fliplr(image).copy()
            # CRITICAL: swap masks when flipping — right parotid moves to left side
            new_mask_r = np.fliplr(mask_l).copy()
            new_mask_l = np.fliplr(mask_r).copy()
            mask_r = new_mask_r
            mask_l = new_mask_l

        return image, mask_r, mask_l

    def get_sample_weights(self):
        parotid_json = os.path.join(
            os.path.dirname(self.dataset_path), 
            'parotid_train_patients.json'
        )
        
        if os.path.exists(parotid_json):
            print("Loading parotid patient list for weights...")
            with open(parotid_json) as f:
                import json
                parotid_patients = set(json.load(f))
            
            weights = []
            for filepath in self.slices:
                parts = filepath.replace(self.dataset_path, '').strip('/').split('/')
                patient_id = parts[0]
                weights.append(14.0 if patient_id in parotid_patients else 1.0)
            
            print(f"Weights assigned for {len(self.slices)} slices instantly.")
            return weights
        
        # Fallback — scan files (slow)
        print(f"Computing sample weights by scanning files...")
        weights = []
        for i, filepath in enumerate(self.slices):
            if i % 2000 == 0:
                print(f"  {i}/{len(self.slices)}", flush=True)
            try:
                with np.load(filepath) as data:
                    has_parotid = ('PAROTID_R' in data.files or 'PAROTID_L' in data.files)
                weights.append(14.0 if has_parotid else 1.0)
            except:
                weights.append(1.0)
        return weights

    def __getitem__(self, idx):
        filepath = self.slices[idx]
        data = np.load(filepath)

        image = self.window_and_normalise(data['image'])

        mask = np.zeros((2, 512, 512), dtype=np.float32)
        if 'PAROTID_R' in data.files:
            mask[0] = data['PAROTID_R'].astype(np.float32)
        if 'PAROTID_L' in data.files:
            mask[1] = data['PAROTID_L'].astype(np.float32)

        if self.augment:
            image, mask[0], mask[1] = self.augment_sample(image, mask[0], mask[1])

        image = torch.tensor(image).unsqueeze(0)
        mask  = torch.tensor(mask)

        return image, mask


if __name__ == '__main__':
    dataset_path = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'
    split_path   = '/Users/ritvikmod/Desktop/ct scan project/dataset_split.json'

    pin_memory = torch.cuda.is_available()

    train_ds = ParotidDataset('train', dataset_path, split_path, augment=True)
    val_ds   = ParotidDataset('val',   dataset_path, split_path, augment=False)
    test_ds  = ParotidDataset('test',  dataset_path, split_path, augment=False)

    train_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler,
                              num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)

    # Sanity check
    print("\nLoading one batch...")
    images, masks = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")
    print(f"Mask batch shape  : {masks.shape}")
    print(f"Image range       : {images.min():.3f} to {images.max():.3f}")
    print(f"Mask unique values: {masks.unique()}")
    print(f"Slices with parotid: {((masks.sum(dim=(1,2,3)) > 0).sum().item())} / {len(masks)}")

    # Visualisation with correct HU
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row = i // 4
        col = i % 4

        img_display = images[i, 0].numpy()

        axes[row, col].imshow(img_display, cmap='gray', vmin=0, vmax=1)

        mask_r = masks[i, 0].numpy()
        if mask_r.max() > 0:
            axes[row, col].imshow(
                np.ma.masked_where(mask_r == 0, mask_r),
                cmap='Reds', alpha=0.6, vmin=0, vmax=1
            )

        mask_l = masks[i, 1].numpy()
        if mask_l.max() > 0:
            axes[row, col].imshow(
                np.ma.masked_where(mask_l == 0, mask_l),
                cmap='Blues', alpha=0.6, vmin=0, vmax=1
            )

        has_p = masks[i].sum() > 0
        axes[row, col].set_title(f"Slice {i} {'[PAROTID]' if has_p else ''}", fontsize=9)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('/Users/ritvikmod/Desktop/ct scan project/dataloader_check_final.png', dpi=150)
    print("Visualisation saved to dataloader_check_final.png")
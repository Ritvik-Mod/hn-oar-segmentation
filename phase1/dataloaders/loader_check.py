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
        self.hu_intercept = -8192.0
        self.hu_slope = 1.0
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
        image_hu = image.astype(np.float32) * self.hu_slope + self.hu_intercept
        image_clipped = np.clip(image_hu, self.hu_min, self.hu_max)
        image_norm = (image_clipped - self.hu_min) / (self.hu_max - self.hu_min)
        return image_norm

    def augment_sample(self, image, mask_r, mask_l):
        do_hflip = random.random() > 0.5
        if do_hflip:
            image = np.fliplr(image).copy()
            # CRITICAL: Swapping the anatomical masks during a horizontal flip
            new_mask_r = np.fliplr(mask_l).copy()
            new_mask_l = np.fliplr(mask_r).copy()
            mask_r = new_mask_r
            mask_l = new_mask_l
        return image, mask_r, mask_l, do_hflip # Now returning the flip status

    def get_sample_weights(self):
        print("Computing sample weights...")
        weights = []
        for filepath in self.slices:
            try:
                data = np.load(filepath)
                has_parotid = ('PAROTID_R' in data.files or 'PAROTID_L' in data.files)
                weights.append(14.0 if has_parotid else 1.0)
            except:
                weights.append(1.0)
        return weights

    def __getitem__(self, idx):
        filepath = self.slices[idx]
        data = np.load(filepath)

        # 1. Get the Raw Dataset Ground Truth
        raw_has_r = bool('PAROTID_R' in data.files)
        raw_has_l = bool('PAROTID_L' in data.files)
        did_hflip = False

        image = self.window_and_normalise(data['image'])

        mask = np.zeros((2, 512, 512), dtype=np.float32)
        if raw_has_r:
            mask[0] = data['PAROTID_R'].astype(np.float32)
        if raw_has_l:
            mask[1] = data['PAROTID_L'].astype(np.float32)

        # 2. Apply Augmentations and track if a flip occurred
        if self.augment:
            image, mask[0], mask[1], did_hflip = self.augment_sample(image, mask[0], mask[1])

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask)

        # Return EVERYTHING for verification
        return image, mask, filepath, raw_has_r, raw_has_l, did_hflip


if __name__ == '__main__':
    dataset_path = '/Users/ritvikmod/Desktop/ct scan project/ML_Dataset_Final'
    split_path   = '/Users/ritvikmod/Desktop/ct scan project/dataset_split.json'
    output_dir   = '/Users/ritvikmod/Desktop/ct scan project/loader testing'

    os.makedirs(output_dir, exist_ok=True)
    pin_memory = torch.cuda.is_available()

    train_ds = ParotidDataset('train', dataset_path, split_path, augment=True)
    train_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler,
                              num_workers=4, pin_memory=pin_memory)

    NUM_BATCHES = 10 

    print(f"\nSaving {NUM_BATCHES} batches to: {output_dir}\n")

    for batch_idx, (images, masks, filepaths, raw_r_flags, raw_l_flags, flip_flags) in enumerate(train_loader):
        if batch_idx >= NUM_BATCHES:
            break

        parotid_count = (masks.sum(dim=(1, 2, 3)) > 0).sum().item()
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx + 1:02d} — {parotid_count}/8 slices have parotid")
        print(f"{'='*70}")
        
        for i in range(8):
            patient_id = filepaths[i].split('/')[-4] if '1~CT1' in filepaths[i] else filepaths[i].split('/')[-3]
            slice_name = filepaths[i].split('/')[-1].replace('.npz', '')
            
            # Extract Raw File Info
            file_r = "YES" if raw_r_flags[i].item() else "NO "
            file_l = "YES" if raw_l_flags[i].item() else "NO "
            flipped = "YES" if flip_flags[i].item() else "NO "
            
            # Extract Final Tensor Info
            pixels_r = int(masks[i, 0].sum().item())
            pixels_l = int(masks[i, 1].sum().item())
            tensor_r = "YES" if pixels_r > 0 else "NO "
            tensor_l = "YES" if pixels_l > 0 else "NO "
            
            print(f"Slice {i} | Patient: {patient_id} | Z-pos: {slice_name}")
            print(f"   [Action] Horizontal Flip Applied: {flipped}")
            print(f"   RAW FILE (Right) : {file_r}  -->  FINAL TENSOR (Ch 0 Red)  : {tensor_r} ({pixels_r} px)")
            print(f"   RAW FILE (Left)  : {file_l}  -->  FINAL TENSOR (Ch 1 Blue) : {tensor_l} ({pixels_l} px)")
            print("-" * 70)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Batch {batch_idx + 1} — {parotid_count}/8 slices have parotid", fontsize=13)

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

            patient_id = filepaths[i].split('/')[-4] if '1~CT1' in filepaths[i] else filepaths[i].split('/')[-3]
            slice_name = filepaths[i].split('/')[-1].replace('.npz', '')
            
            flipped_tag = "[FLIPPED]" if flip_flags[i].item() else ""
            has_p = masks[i].sum() > 0
            
            title = f"{patient_id}\n{slice_name} {flipped_tag}"
            if has_p:
                title += "\n[PAROTID]"
            axes[row, col].set_title(title, fontsize=7)
            axes[row, col].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"batch_{batch_idx + 1:02d}.png")
        plt.savefig(save_path, dpi=130)
        plt.close()

    print(f"\nDone. {NUM_BATCHES} images saved to {output_dir}")
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import h5py

class ParotidDataset(Dataset):
    def __init__(self, split, h5_path, split_json_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self.hu_intercept = -8192.0
        self.hu_slope = 1.0
        self.hu_min = -150.0
        self.hu_max = 250.0

        with open(split_json_path) as f:
            self.patients = json.load(f)[split]

        self.slices = []
        self.slice_has_mask = []
        print(f"Scanning HDF5 vault for {split} split and mapping masks (takes ~60 seconds)...")
        
        # visititems gives us both the folder name AND the object inside it so we can check for masks instantly
        def collect_slices(name, obj):
            if isinstance(obj, h5py.Group) and name.endswith('.npz'):
                patient_id = name.split('/')[0]
                if patient_id in self.patients:
                    self.slices.append(name)
                    # Check if either parotid gland exists in this specific slice
                    has_mask = ('PAROTID_R' in obj) or ('PAROTID_L' in obj)
                    self.slice_has_mask.append(has_mask)

        with h5py.File(self.h5_path, 'r') as hf:
            hf.visititems(collect_slices)

        mask_count = sum(self.slice_has_mask)
        print(f"{split}: {len(self.slices)} total slices found. {mask_count} contain Parotid glands.")
        self.h5_file = None 

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
            new_mask_r = np.fliplr(mask_l).copy()
            new_mask_l = np.fliplr(mask_r).copy()
            mask_r, mask_l = new_mask_r, new_mask_l
        return image, mask_r, mask_l

    def get_sample_weights(self):
        print("Assigning slice-level weights (Empty=1.0, Mask=20.0)...")
        # Oversample slices with actual glands so the model stops guessing 'background'
        weights = [20.0 if has_mask else 1.0 for has_mask in self.slice_has_mask]
        return weights

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        key = self.slices[idx]
        grp = self.h5_file[key]
        image = self.window_and_normalise(grp['image'][...])

        mask = np.zeros((2, 512, 512), dtype=np.float32)
        if 'PAROTID_R' in grp:
            mask[0] = grp['PAROTID_R'][...].astype(np.float32)
        if 'PAROTID_L' in grp:
            mask[1] = grp['PAROTID_L'][...].astype(np.float32)

        if self.augment:
            image, mask[0], mask[1] = self.augment_sample(image, mask[0], mask[1])

        image = torch.tensor(image).unsqueeze(0)
        mask  = torch.tensor(mask)
        return image, mask

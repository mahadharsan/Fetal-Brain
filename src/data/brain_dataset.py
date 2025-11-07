"""
BRAIN SEGMENTATION DATASET
==========================
Modified dataset class to extract brain masks (red pixels) instead of ventricles.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import pandas as pd

class BrainSegmentationDataset(Dataset):
    """
    Dataset for brain tissue segmentation (not ventricles).
    Extracts red pixels (brain) from the colored masks.
    """
    
    def __init__(self, 
                 image_dir, 
                 mask_dir,
                 csv_report='matching_report.csv',
                 mode='train',
                 img_size=256,
                 transform=None):
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        
        # Load matched pairs from CSV
        df = pd.read_csv(csv_report)
        matched_df = df[df['status'] == 'matched']
        self.matched_files = matched_df['filename'].tolist()
        
        # Split data (70/15/15) with same seed as ventricle model
        random.seed(42)
        n_total = len(self.matched_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        indices = list(range(n_total))
        random.shuffle(indices)
        
        if mode == 'train':
            self.indices = indices[:n_train]
        elif mode == 'val':
            self.indices = indices[n_train:n_train+n_val]
        else:  # test
            self.indices = indices[n_train+n_val:]
        
        print(f"Brain Dataset {mode}: {len(self.indices)} images")
    
    def __len__(self):
        return len(self.indices)
    
    def extract_brain_mask(self, mask_img):
        """
        Extract brain tissue pixels (red in BGR format).
        """
        brain_mask = (
            (mask_img[:,:,0] == 0) &    # Blue = 0
            (mask_img[:,:,1] == 0) &    # Green = 0
            (mask_img[:,:,2] == 255)    # Red = 255
        )
        return brain_mask.astype(np.float32)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        filename = self.matched_files[actual_idx]
        
        # Load ultrasound
        img_path = self.image_dir / f"{filename}.png"
        image = cv2.imread(str(img_path))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load colored mask
        mask_path = self.mask_dir / f"{filename}.png"
        mask_img = cv2.imread(str(mask_path))
        
        # Extract BRAIN mask (not ventricles)
        brain_mask = self.extract_brain_mask(mask_img)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        brain_mask = cv2.resize(brain_mask, (self.img_size, self.img_size))
        
        # Augmentation for training
        if self.transform and self.mode == 'train':
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                brain_mask = np.fliplr(brain_mask).copy()
        
        # Normalize and convert to tensors
        image = image.astype(np.float32) / 255.0
        brain_mask = (brain_mask > 0.5).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        brain_mask = torch.from_numpy(brain_mask).unsqueeze(0)
        
        return {
            'image': image,
            'mask': brain_mask,
            'img_name': filename
        }
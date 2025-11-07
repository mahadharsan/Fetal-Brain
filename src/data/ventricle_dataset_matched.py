"""
VENTRICLE DATASET - MATCHED PAIRS ONLY
=======================================
This version only loads the 584 images that have matching masks.
No more guessing or wrong plane matching!

Original Files:
├── Ultrasound: Grayscale medical image
└── Mask: Colored regions (red=brain, blue=ventricles)
           ↓
Dataset Does This:
├── Step 1: Load both files
├── Step 2: Extract blue parts only (ventricles)
├── Step 3: Calculate size (23% of brain)
├── Step 4: Classify (Normal/Mild/Severe)
└── Step 5: Package for AI model
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import pandas as pd

class VentricleDatasetMatched(Dataset):
    """
    Improved dataset that only uses verified matched pairs.
    """
    
    def __init__(self, 
                 image_dir, 
                 mask_dir,
                 csv_report='matching_report.csv',
                 mode='train',
                 img_size=256,
                 transform=None):
        """
        Parameters:
        -----------
        image_dir : str
            Path to ultrasound images
        mask_dir : str  
            Path to segmentation masks
        csv_report : str
            Path to the matching report CSV (from find_matching_pairs.py)
        mode : str
            'train', 'val', or 'test'
        img_size : int
            Size to resize images to
        transform : bool
            Whether to apply augmentations
        """
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        
        # Load the matched pairs from CSV
        print(f"Loading matched pairs from {csv_report}")
        df = pd.read_csv(csv_report)
        
        # Filter to only matched pairs
        matched_df = df[df['status'] == 'matched']
        print(f"Found {len(matched_df)} matched pairs")
        
        # Get list of matched filenames
        self.matched_files = matched_df['filename'].tolist()
        
        # Split into train/val/test (70/15/15)
        random.seed(42)
        n_total = len(self.matched_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        # Shuffle for random split
        indices = list(range(n_total))
        random.shuffle(indices)
        
        if mode == 'train':
            self.indices = indices[:n_train]
        elif mode == 'val':
            self.indices = indices[n_train:n_train+n_val]
        else:  # test
            self.indices = indices[n_train+n_val:]
        
        print(f"Dataset {mode}: {len(self.indices)} images")
        
        # Color map for extracting ventricles
        self.color_map = {
            'background': [0, 0, 0],
            'brain': [0, 0, 255],      # Red in BGR
            'csp': [0, 255, 0],        # Green
            'lv': [255, 0, 0]          # Blue in BGR (ventricles)
        }
    
    def __len__(self):
        return len(self.indices)
    
    def extract_ventricle_mask(self, mask_img):
        """Extract ventricle pixels (blue in BGR format)"""
        ventricle_mask = (
            (mask_img[:,:,0] == 255) &  # B=255
            (mask_img[:,:,1] == 0) &     # G=0  
            (mask_img[:,:,2] == 0)       # R=0
        )
        return ventricle_mask.astype(np.float32)
    
    def extract_brain_mask(self, mask_img):
        """Extract brain pixels (red in BGR format)"""
        brain_mask = (
            (mask_img[:,:,0] == 0) &    # B=0
            (mask_img[:,:,1] == 0) &    # G=0
            (mask_img[:,:,2] == 255)    # R=255
        )
        return brain_mask.astype(np.float32)
    
    def calculate_vhr(self, ventricle_mask, brain_mask):
        """Calculate Ventricle-to-Hemisphere Ratio"""
        ventricle_area = np.sum(ventricle_mask)
        brain_area = np.sum(brain_mask)
        
        if (brain_area + ventricle_area) == 0:
            return 0.0
            
        vhr = (ventricle_area / (brain_area + ventricle_area)) * 100
        return vhr
    
    def __getitem__(self, idx):
        """Load one matched pair"""
        
        # Get filename for this index
        actual_idx = self.indices[idx]
        filename = self.matched_files[actual_idx]
        
        # Build full paths (both have same filename)
        img_path = self.image_dir / f"{filename}.png"
        mask_path = self.mask_dir / f"{filename}.png"
        
        # Verify both exist
        if not img_path.exists() or not mask_path.exists():
            print(f"Warning: Missing file for {filename}")
            # Return a dummy sample
            return {
                'image': torch.zeros(1, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'vhr': torch.tensor(0.0),
                'severity': torch.tensor(0),
                'img_name': filename
            }
        
        # Load ultrasound
        image = cv2.imread(str(img_path))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load mask  
        mask_img = cv2.imread(str(mask_path))
        
        # Extract ventricles and brain
        ventricle_mask = self.extract_ventricle_mask(mask_img)
        brain_mask = self.extract_brain_mask(mask_img)
        
        # Calculate VHR
        vhr = self.calculate_vhr(ventricle_mask, brain_mask)
        
        # Classify severity based on VHR
        if vhr < 25:
            severity = 0  # Normal
        elif vhr < 35:
            severity = 1  # Mild
        elif vhr < 50:
            severity = 2  # Moderate  
        else:
            severity = 3  # Severe
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        ventricle_mask = cv2.resize(ventricle_mask, (self.img_size, self.img_size))
        
        # Simple augmentation for training
        if self.transform and self.mode == 'train':
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                ventricle_mask = np.fliplr(ventricle_mask).copy()
        
        # Normalize and convert to tensors
        image = image.astype(np.float32) / 255.0
        ventricle_mask = (ventricle_mask > 0.5).astype(np.float32)  # Ensure binary after resize
        
        image = torch.from_numpy(image).unsqueeze(0)
        ventricle_mask = torch.from_numpy(ventricle_mask).unsqueeze(0)
        
        return {
            'image': image,
            'mask': ventricle_mask,
            'vhr': torch.tensor(vhr, dtype=torch.float32),
            'severity': torch.tensor(severity, dtype=torch.long),
            'img_name': filename
        }

# TEST THE DATASET
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MATCHED DATASET")
    print("=" * 60)
    
    # Create dataset
    dataset = VentricleDatasetMatched(
        image_dir='../../data/raw/trans_ventricular_original',
        mask_dir='../../data/raw/trans_ventricular/segmentation_mask/SegmentationClass',
        csv_report='matching_report.csv', 
        mode='train'
    )
    
    print(f"\n✅ Dataset ready!")
    print(f"   Training samples: {len(dataset)}")
    
    # Test loading a few samples
    print("\n📊 Testing first 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   VHR: {sample['vhr'].item():.2f}%")
        print(f"   Severity: {sample['severity'].item()} ", end="")
        severity_names = ['Normal', 'Mild', 'Moderate', 'Severe']
        print(f"({severity_names[sample['severity'].item()]})")
        print(f"   File: {sample['img_name']}")
    
    print("\n✅ Dataset test complete!")
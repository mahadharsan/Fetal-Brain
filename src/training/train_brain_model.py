"""
TRAIN BRAIN SEGMENTATION MODEL
==============================
Quick training script for brain segmentation.
Uses same architecture as ventricle model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet import UNet
from src.data.brain_dataset import BrainSegmentationDataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 20  # Fewer epochs since brain is easier to segment
IMG_SIZE = 256

# Paths
DATA_DIR = Path('data/raw')
IMAGE_DIR = DATA_DIR / 'trans_ventricular_original'
MASK_DIR = DATA_DIR / 'trans_ventricular/segmentation_mask/SegmentationClass'
CSV_REPORT = Path('matching_report.csv')

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def train_brain_model():
    print("="*60)
    print("BRAIN SEGMENTATION MODEL TRAINING")
    print("="*60)
    
    # Create datasets
    print("\n1. Loading datasets...")
    train_dataset = BrainSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        csv_report=CSV_REPORT,
        mode='train',
        img_size=IMG_SIZE,
        transform=True
    )
    
    val_dataset = BrainSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        csv_report=CSV_REPORT,
        mode='val',
        img_size=IMG_SIZE,
        transform=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\n2. Creating model...")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    print(f"   Model on {DEVICE}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n3. Training...")
    best_dice = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)
            dice = dice_coefficient(predictions, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                
                predictions = model(images)
                loss = criterion(predictions, masks)
                dice = dice_coefficient(predictions, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
        
        # Calculate averages
        avg_train_dice = train_dice / len(train_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"\nEpoch {epoch+1}: Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'brain_segmentation_model.pth')
            print(f"   ✓ Saved new best model (Dice: {best_dice:.4f})")
    
    print("\n" + "="*60)
    print(f"Training complete! Best validation Dice: {best_dice:.4f}")
    print("Model saved as: brain_segmentation_model.pth")
    print("="*60)

if __name__ == "__main__":
    train_brain_model()
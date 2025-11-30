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


def main():
    """
    Main training function for Brain Segmentation.
    """
    import matplotlib.pyplot as plt
    print("="*60)
    print("BRAIN SEGMENTATION TRAINING")
    print("="*60)
    
    # CREATE DATASETS
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
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # CREATE DATA LOADERS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # CREATE MODEL
    print("\n2. Creating model...")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    print(f"   Model on {DEVICE}")
    
    # LOSS FUNCTION AND OPTIMIZER
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # TRAINING LOOP
    print("\n3. Starting training...")
    print("-"*40)
    best_dice = 0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            predictions = model(images)
            loss = criterion(predictions, masks)
            dice = dice_coefficient(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
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
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        
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
        
    # PLOT TRAINING HISTORY
    print("\n4. Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_dices, label='Train Dice')
    ax2.plot(val_dices, label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('brain_training_history.png')
    plt.show()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Model saved as: brain_segmentation_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()

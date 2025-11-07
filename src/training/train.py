"""
TRAINING SCRIPT FOR VENTRICLE SEGMENTATION
===========================================
This script trains the U-Net to identify ventricles in fetal brain ultrasounds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import our modules
sys.path.append('..')

# Import our custom modules
from src.models.unet import UNet
from src.data.ventricle_dataset_matched import VentricleDatasetMatched

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# HYPERPARAMETERS
# ---------------
# These are settings that control how the model learns
BATCH_SIZE = 8          # Number of images to process at once (reduce if GPU memory error)
LEARNING_RATE = 0.001   # How fast the model learns (too high = unstable, too low = slow)
NUM_EPOCHS = 50         # How many times to go through all training data
IMG_SIZE = 256          # Image dimensions (256x256)

# DATA PATHS
# ----------
DATA_DIR = Path('data/raw')
IMAGE_DIR = DATA_DIR / 'trans_ventricular_original'
MASK_DIR = DATA_DIR / 'trans_ventricular/segmentation_mask/SegmentationClass'
CSV_REPORT = Path('matching_report.csv')

print(f"Checking paths exist:")
print(f"  IMAGE_DIR: {IMAGE_DIR.exists()} - {IMAGE_DIR}")
print(f"  MASK_DIR: {MASK_DIR.exists()} - {MASK_DIR}")
print(f"  CSV_REPORT: {CSV_REPORT.exists()} - {CSV_REPORT}")

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient (measure of overlap).
    
    Dice = 2 * (Prediction ∩ Target) / (Prediction + Target)
    
    Perfect overlap = 1.0
    No overlap = 0.0
    
    Parameters:
    -----------
    pred : torch.Tensor
        Predicted mask
    target : torch.Tensor
        Ground truth mask
    smooth : float
        Small value to avoid division by zero
    
    Returns:
    --------
    float
        Dice coefficient (0-1)
    """
    pred = torch.sigmoid(pred)  # Convert to probabilities
    pred = (pred > 0.5).float()  # Convert to binary
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    Train the model for one epoch (one pass through all training data).
    
    Parameters:
    -----------
    model : nn.Module
        The U-Net model
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim
        Optimizer for updating weights
    epoch : int
        Current epoch number
    
    Returns:
    --------
    float, float
        Average loss and dice score for the epoch
    """
    model.train()  # Set model to training mode
    
    total_loss = 0
    total_dice = 0
    
    # Progress bar for visual feedback
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    
    for batch in pbar:
        # Move data to GPU
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, masks)
        
        # Calculate Dice score for monitoring
        dice = dice_coefficient(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        total_loss += loss.item()
        total_dice += dice.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    
    return avg_loss, avg_dice

def validate(model, val_loader, criterion):
    """
    Validate the model on validation data (no training).
    
    Parameters:
    -----------
    model : nn.Module
        The U-Net model
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    
    Returns:
    --------
    float, float
        Average loss and dice score for validation
    """
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():  # Don't calculate gradients (saves memory)
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            predictions = model(images)
            
            loss = criterion(predictions, masks)
            dice = dice_coefficient(predictions, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    
    return avg_loss, avg_dice

def visualize_prediction(model, dataset, device, num_samples=3):
    """
    Visualize model predictions on random samples.
    
    Shows: Original image | True mask | Predicted mask
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        
        image = sample['image'].unsqueeze(0).to(device)
        true_mask = sample['mask'].squeeze().cpu().numpy()
        
        # Predict
        with torch.no_grad():
            prediction = model(image)
            pred_mask = torch.sigmoid(prediction)
            pred_mask = (pred_mask > 0.5).float()
            pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Ultrasound')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask, cmap='hot')
        axes[i, 1].set_title('True Ventricles')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='hot')
        axes[i, 2].set_title('Predicted Ventricles')
        axes[i, 2].axis('off')
        
        # Calculate Dice for this sample
        dice = np.sum(pred_mask * true_mask) * 2 / (np.sum(pred_mask) + np.sum(true_mask) + 1e-6)
        fig.text(0.9, 0.8 - i*0.27, f'Dice: {dice:.3f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def main():
    """
    Main training function.
    """
    print("="*60)
    print("VENTRICLE SEGMENTATION TRAINING")
    print("="*60)
    
    # CREATE DATASETS
    # ---------------
    print("\n1. Loading datasets...")
    
    train_dataset = VentricleDatasetMatched(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        csv_report=CSV_REPORT,
        mode='train',
        img_size=IMG_SIZE,
        transform=True
    )
    
    val_dataset = VentricleDatasetMatched(
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
    # -------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Randomize order each epoch
        num_workers=0  # Set to 0 for Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # CREATE MODEL
    # ------------
    print("\n2. Creating model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    print(f"   Model on {device}")
    
    # LOSS FUNCTION AND OPTIMIZER
    # ---------------------------
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (reduces LR when stuck)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # TRAINING LOOP
    # -------------
    print("\n3. Starting training...")
    print("-"*40)
    
    best_dice = 0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion)
        
        # Adjust learning rate
        scheduler.step(val_dice)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'best_model.pth')
            print(f"   ✓ Saved new best model (Dice: {best_dice:.4f})")
        
        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\n   Generating prediction samples...")
            visualize_prediction(model, val_dataset, device)
    
    # PLOT TRAINING HISTORY
    # ---------------------
    print("\n4. Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice plot
    ax2.plot(train_dices, label='Train Dice')
    ax2.plot(val_dices, label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Model saved as: best_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()
"""
U-NET MODEL FOR VENTRICLE SEGMENTATION
=======================================
U-Net is a popular architecture for medical image segmentation.
It looks like the letter U - goes down (encoding) then up (decoding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net architecture for segmenting ventricles in fetal brain ultrasounds.
    
    The network has a U-shape:
    - Left side (encoder): Compresses image to learn features
    - Bottom: Most compressed representation
    - Right side (decoder): Expands back to original size
    - Skip connections: Connect left to right to preserve details
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize U-Net.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels (1 for grayscale ultrasound)
        out_channels : int
            Number of output channels (1 for binary ventricle mask)
        """
        super(UNet, self).__init__()
        
        # ENCODER (Downsampling path - left side of U)
        # Each block doubles the number of features and halves the spatial size
        
        # Block 1: 1 -> 64 channels
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # Reduce size by half
        
        # Block 2: 64 -> 128 channels  
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3: 128 -> 256 channels
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        # Block 4: 256 -> 512 channels
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        
        # BOTTOM of U (Most compressed)
        # Block 5: 512 -> 1024 channels
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        
        # DECODER (Upsampling path - right side of U)
        # Each block halves features and doubles spatial size
        
        # Upblock 4: 1024 -> 512 channels
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # 1024 because of concatenation
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Upblock 3: 512 -> 256 channels
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Upblock 2: 256 -> 128 channels
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Upblock 1: 128 -> 64 channels
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final output layer: 64 -> 1 channel (ventricle mask)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, height, width)
            
        Returns:
        --------
        torch.Tensor
            Predicted mask of shape (batch, 1, height, width)
        """
        
        # ENCODER PATH (going down the U)
        # Each block: Conv -> ReLU -> Conv -> ReLU -> Pool
        
        # Block 1
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))
        x1 = self.bn1(x1)  # Batch norm
        x1_pool = self.pool1(x1)  # Save x1 for skip connection
        
        # Block 2
        x2 = F.relu(self.conv2_1(x1_pool))
        x2 = F.relu(self.conv2_2(x2))
        x2 = self.bn2(x2)
        x2_pool = self.pool2(x2)
        
        # Block 3
        x3 = F.relu(self.conv3_1(x2_pool))
        x3 = F.relu(self.conv3_2(x3))
        x3 = self.bn3(x3)
        x3_pool = self.pool3(x3)
        
        # Block 4
        x4 = F.relu(self.conv4_1(x3_pool))
        x4 = F.relu(self.conv4_2(x4))
        x4 = self.bn4(x4)
        x4_pool = self.pool4(x4)
        
        # BOTTOM (no pooling here)
        x5 = F.relu(self.conv5_1(x4_pool))
        x5 = F.relu(self.conv5_2(x5))
        x5 = self.bn5(x5)
        
        # DECODER PATH (going up the U)
        # Each block: Upsample -> Concatenate with encoder -> Conv -> ReLU -> Conv -> ReLU
        
        # Upblock 4
        up4 = self.upconv4(x5)
        up4 = torch.cat([up4, x4], dim=1)  # Skip connection from encoder
        up4 = F.relu(self.conv6_1(up4))
        up4 = F.relu(self.conv6_2(up4))
        
        # Upblock 3
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, x3], dim=1)
        up3 = F.relu(self.conv7_1(up3))
        up3 = F.relu(self.conv7_2(up3))
        
        # Upblock 2
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, x2], dim=1)
        up2 = F.relu(self.conv8_1(up2))
        up2 = F.relu(self.conv8_2(up2))
        
        # Upblock 1
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, x1], dim=1)
        up1 = F.relu(self.conv9_1(up1))
        up1 = F.relu(self.conv9_2(up1))
        
        # Final output layer
        out = self.conv_out(up1)
        
        # Note: We don't apply sigmoid here because we'll use BCEWithLogitsLoss
        # which combines sigmoid and BCE for numerical stability
        
        return out

# TEST THE MODEL
if __name__ == "__main__":
    """
    Test that the U-Net works and outputs correct shape.
    """
    print("=" * 60)
    print("TESTING U-NET MODEL")
    print("=" * 60)
    
    # Create model
    model = UNet(in_channels=1, out_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test with dummy input
    batch_size = 2
    height = 256
    width = 256
    
    dummy_input = torch.randn(batch_size, 1, height, width)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Verify output shape matches input shape
    assert output.shape == dummy_input.shape, "Output shape doesn't match input!"
    
    print("\n✅ U-Net test passed! Model is ready for training.")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"\n🚀 GPU available: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        output = model(dummy_input)
        print("   Model successfully moved to GPU!")
    else:
        print("\n⚠️  No GPU available, will train on CPU (slower)")
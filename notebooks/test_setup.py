import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Create a dummy image to test OpenCV
test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
print("Test image shape:", test_image.shape)
plt.imshow(test_image, cmap='gray')
plt.title("Test Image")
plt.show()
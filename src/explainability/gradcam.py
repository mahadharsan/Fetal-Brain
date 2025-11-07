import torch
import numpy as np
import cv2

class GradCAM:
    """Generate visual explanations for model predictions."""
    
    def __init__(self, model, target_layer_name='down4'):
        self.model = model
        self.model.eval()
        
        # Get the target layer
        self.target_layer = self._get_layer(model, target_layer_name)
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handle_forward = self.target_layer.register_forward_hook(self.save_activation)
        self.handle_backward = self.target_layer.register_backward_hook(self.save_gradient)
    
    def _get_layer(self, model, layer_name):
        """Get layer by name."""
        return getattr(model, layer_name)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image):
        """Generate attention heatmap."""
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Target the positive predictions (ventricles)
        target = torch.sigmoid(output).sum()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Generate heatmap
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Weight the activations
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # Clean up
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (256, 256))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def create_visualization(self, original_image, heatmap, alpha=0.4):
        """Overlay heatmap on image."""
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), 
            cv2.COLORMAP_JET
        )
        
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(
                (original_image * 255).astype(np.uint8), 
                cv2.COLOR_GRAY2RGB
            )
        else:
            original_rgb = (original_image * 255).astype(np.uint8)
        
        # Overlay
        superimposed = cv2.addWeighted(
            original_rgb, 1-alpha, 
            heatmap_colored, alpha, 0
        )
        
        return superimposed
    
    def cleanup(self):
        """Remove hooks."""
        self.handle_forward.remove()
        self.handle_backward.remove()
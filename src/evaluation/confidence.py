import numpy as np
import cv2
import torch

class ConfidenceEstimator:
    """Calculate confidence scores for segmentation predictions."""
    
    def calculate_segmentation_confidence(self, pred_tensor, mask, model_type='ventricle'):
        """
        Calculate confidence for a segmentation prediction.
        
        Args:
            pred_tensor: Raw model output (before sigmoid)
            mask: Binary mask (after thresholding)
            model_type: 'ventricle' or 'brain' for specific checks
        """
        with torch.no_grad():
            # Get probability map
            prob_map = torch.sigmoid(pred_tensor).cpu().numpy().squeeze()
            
            # 1. Prediction certainty (how far from 0.5)
            certainty = np.abs(prob_map - 0.5) * 2
            avg_certainty = np.mean(certainty) * 100
            
            # 2. Size validation
            pixel_count = np.sum(mask)
            size_confidence = self._validate_size(pixel_count, model_type)
            
            # 3. Edge sharpness
            edge_confidence = self._calculate_edge_confidence(prob_map, mask)
            
            # Combined confidence
            overall = (avg_certainty * 0.4 + size_confidence * 0.3 + edge_confidence * 0.3)
            
            return {
                'overall': overall,
                'certainty': avg_certainty,
                'size_validity': size_confidence,
                'edge_quality': edge_confidence,
                'reliable': overall > 70,
                'warning': overall < 50
            }
    
    def _validate_size(self, pixel_count, model_type):
        """Check if detected size is reasonable."""
        if model_type == 'ventricle':
            if pixel_count < 10:
                return 10  # Too small
            elif pixel_count > 2000:
                return 50  # Suspiciously large
            else:
                return 100  # Good size
        else:  # brain
            if pixel_count < 1000:
                return 20
            elif pixel_count > 50000:
                return 60
            else:
                return 100
    
    def _calculate_edge_confidence(self, prob_map, mask):
        """Assess boundary clarity."""
        edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
        if not edges.any():
            return 0
        
        edge_pixels = edges > 0
        edge_probs = prob_map[edge_pixels]
        # Clear edges have probs far from 0.5
        edge_clarity = np.mean(np.abs(edge_probs - 0.5)) * 200
        return min(edge_clarity, 100)
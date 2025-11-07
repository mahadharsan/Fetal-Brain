"""
END-TO-END VENTRICLE MEASUREMENT PIPELINE
==========================================
This version uses ONLY the ultrasound image and model predictions.
No ground truth masks are used during inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VentricleMeasurement:
    """Data class to store ventricle measurements."""
    ventricle_area_pixels: int
    brain_area_pixels: int
    total_area_pixels: int
    vhr_percentage: float
    ventricle_area_mm2: float
    equivalent_diameter_mm: float
    severity_classification: str
    severity_score: int
    confidence: float


class EndToEndAnalyzer:
    """
    Analyzes ventricles using ONLY ultrasound images and model predictions.
    No ground truth data is used during inference.
    """
    
    # Clinical thresholds
    VHR_THRESHOLDS = {
        'normal': 25.0,
        'mild': 35.0,
        'moderate': 50.0,
    }
    
    # Pixel spacing (would come from DICOM in real system)
    DEFAULT_PIXEL_SPACING_MM = 0.5
    
    def __init__(self, pixel_spacing_mm: float = None):
        self.pixel_spacing_mm = pixel_spacing_mm or self.DEFAULT_PIXEL_SPACING_MM
        logger.info(f"Initialized end-to-end analyzer with pixel spacing: {self.pixel_spacing_mm} mm/pixel")
    
    def estimate_brain_region_from_ultrasound(self, ultrasound_image: np.ndarray) -> np.ndarray:
        """
        Estimate brain region from ultrasound WITHOUT using any ground truth.
        
        This uses image processing to find the brain area:
        1. Apply threshold to remove black background
        2. Find largest connected component (the head)
        3. Fill holes to get complete brain region
        
        Parameters:
        -----------
        ultrasound_image : np.ndarray
            Grayscale ultrasound image (0-1 range or 0-255)
            
        Returns:
        --------
        np.ndarray
            Binary mask of estimated brain region
        """
        # Ensure image is in 0-255 range
        if ultrasound_image.max() <= 1.0:
            ultrasound_image = (ultrasound_image * 255).astype(np.uint8)
        else:
            ultrasound_image = ultrasound_image.astype(np.uint8)
        
        # Step 1: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(ultrasound_image, (5, 5), 0)
        
        # Step 2: Apply adaptive threshold to find tissue
        # Use a low threshold to capture the dark brain tissue
        _, binary = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
        
        # Step 3: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Step 4: Find largest connected component (the head)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background which is label 0)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            brain_mask = (labels == largest_component).astype(np.uint8)
        else:
            # If no components found, use the whole non-black area
            brain_mask = binary > 0
        
        # Step 5: Fill holes in the brain mask
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        # Convert to binary
        brain_mask = brain_mask.astype(np.uint8)
        
        return brain_mask
    
    def measure_ventricles(self, 
                          ventricle_mask: np.ndarray,
                          ultrasound_image: np.ndarray) -> VentricleMeasurement:
        """
        Extract measurements using ONLY the ultrasound and predicted ventricle mask.
        
        Parameters:
        -----------
        ventricle_mask : np.ndarray
            Binary mask of ventricles predicted by the model
        ultrasound_image : np.ndarray
            Original ultrasound image
            
        Returns:
        --------
        VentricleMeasurement
            Complete measurement results
        """
        # Ensure binary ventricle mask
        ventricle_mask = (ventricle_mask > 0.5).astype(np.uint8)
        
        # Estimate brain region from ultrasound (no ground truth!)
        brain_mask = self.estimate_brain_region_from_ultrasound(ultrasound_image)
        
        # Count pixels
        ventricle_pixels = np.sum(ventricle_mask)
        brain_pixels = np.sum(brain_mask)
        total_pixels = ventricle_mask.size
        
        # Calculate VHR
        if brain_pixels > 0:
            vhr = (ventricle_pixels / brain_pixels) * 100
        else:
            vhr = 0.0
        
        # Calculate physical measurements
        pixel_area_mm2 = self.pixel_spacing_mm ** 2
        ventricle_area_mm2 = ventricle_pixels * pixel_area_mm2
        
        # Equivalent diameter
        if ventricle_area_mm2 > 0:
            equivalent_diameter_mm = 2 * np.sqrt(ventricle_area_mm2 / np.pi)
        else:
            equivalent_diameter_mm = 0.0
        
        # Classify severity
        severity_classification, severity_score = self.classify_severity(vhr)
        
        # Calculate confidence
        confidence = self.calculate_confidence(ventricle_mask, brain_mask, ultrasound_image)
        
        return VentricleMeasurement(
            ventricle_area_pixels=int(ventricle_pixels),
            brain_area_pixels=int(brain_pixels),
            total_area_pixels=int(total_pixels),
            vhr_percentage=float(vhr),
            ventricle_area_mm2=float(ventricle_area_mm2),
            equivalent_diameter_mm=float(equivalent_diameter_mm),
            severity_classification=severity_classification,
            severity_score=severity_score,
            confidence=float(confidence)
        )
    
    def classify_severity(self, vhr: float) -> Tuple[str, int]:
        """Classify severity based on VHR."""
        if vhr < self.VHR_THRESHOLDS['normal']:
            return "Normal", 0
        elif vhr < self.VHR_THRESHOLDS['mild']:
            return "Mild Ventriculomegaly", 1
        elif vhr < self.VHR_THRESHOLDS['moderate']:
            return "Moderate Ventriculomegaly", 2
        else:
            return "Severe - Possible Hydrocephalus", 3
    
    def calculate_confidence(self, ventricle_mask: np.ndarray, 
                            brain_mask: np.ndarray,
                            ultrasound_image: np.ndarray) -> float:
        """Calculate confidence in the measurement."""
        confidence_factors = []
        
        # Factor 1: Ventricles present
        ventricle_pixels = np.sum(ventricle_mask)
        if ventricle_pixels > 100:
            confidence_factors.append(1.0)
        elif ventricle_pixels > 10:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Factor 2: Brain region detected
        brain_pixels = np.sum(brain_mask)
        if brain_pixels > 1000:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Factor 3: Image quality (contrast)
        if ultrasound_image.max() > ultrasound_image.min():
            contrast = ultrasound_image.std() / (ultrasound_image.mean() + 1e-6)
            if contrast > 0.3:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    def visualize_analysis(self, 
                          ultrasound: np.ndarray,
                          ventricle_mask: np.ndarray,
                          brain_mask: np.ndarray,
                          measurement: VentricleMeasurement,
                          save_path: Optional[str] = None):
        """Create visualization of the analysis."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original ultrasound
        axes[0].imshow(ultrasound, cmap='gray')
        axes[0].set_title('Original Ultrasound')
        axes[0].axis('off')
        
        # Estimated brain region
        axes[1].imshow(brain_mask, cmap='hot')
        axes[1].set_title('Estimated Brain Region\n(from ultrasound only)')
        axes[1].axis('off')
        
        # Predicted ventricles
        axes[2].imshow(ventricle_mask, cmap='hot')
        axes[2].set_title('Predicted Ventricles\n(from model)')
        axes[2].axis('off')
        
        # Overlay
        overlay = ultrasound.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay[ventricle_mask > 0] = [255, 0, 0]  # Red for ventricles
        axes[3].imshow(overlay)
        axes[3].set_title(f'Result: {measurement.severity_classification}\nVHR: {measurement.vhr_percentage:.1f}%')
        axes[3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self, measurement: VentricleMeasurement) -> str:
        """Generate clinical report."""
        report = []
        report.append("END-TO-END VENTRICLE ANALYSIS")
        report.append("=" * 40)
        report.append("(No ground truth used - fully automated)")
        report.append("")
        
        report.append("MEASUREMENTS:")
        report.append(f"  Ventricle-to-Brain Ratio: {measurement.vhr_percentage:.1f}%")
        report.append(f"  Ventricle Area: {measurement.ventricle_area_mm2:.1f} mm²")
        report.append(f"  Equivalent Diameter: {measurement.equivalent_diameter_mm:.1f} mm")
        report.append("")
        
        report.append("CLASSIFICATION:")
        report.append(f"  {measurement.severity_classification}")
        report.append(f"  Confidence: {measurement.confidence:.0%}")
        report.append("")
        
        report.append("CLINICAL RECOMMENDATION:")
        if measurement.severity_score == 0:
            report.append("  Normal findings. Routine follow-up.")
        elif measurement.severity_score == 1:
            report.append("  Mild enlargement. Follow-up in 2-4 weeks.")
        elif measurement.severity_score == 2:
            report.append("  Moderate enlargement. Consider MRI and specialist referral.")
        else:
            report.append("  URGENT: Severe enlargement. Immediate specialist referral.")
        
        return "\n".join(report)


def run_end_to_end_analysis():
    """
    Run complete end-to-end analysis on test images.
    Uses ONLY the ultrasound and model predictions - no ground truth!
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.models.unet import UNet
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = EndToEndAnalyzer()
    
    # Load trained model
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load('src/training/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get test images (using fixed indices for consistency)
    image_dir = Path('data/raw/trans_ventricular_original')
    all_images = sorted(list(image_dir.glob('*.png')))
    
    # Use deterministic test set (last 10 images)
    test_images = all_images[-10:]
    
    results = []
    print("\n" + "="*60)
    print("END-TO-END ANALYSIS (No Ground Truth)")
    print("="*60)
    
    for idx, img_path in enumerate(test_images[:5], 1):
        print(f"\nAnalyzing Image {idx}: {img_path.name}")
        print("-"*40)
        
        # Load ultrasound
        ultrasound = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ultrasound_resized = cv2.resize(ultrasound, (256, 256))
        
        # Prepare for model
        img_tensor = torch.from_numpy(ultrasound_resized.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict ventricles
        with torch.no_grad():
            prediction = model(img_tensor)
            ventricle_mask = torch.sigmoid(prediction) > 0.5
            ventricle_mask = ventricle_mask.cpu().numpy().squeeze()
        
        # Measure (using ONLY ultrasound and prediction)
        measurement = analyzer.measure_ventricles(
            ventricle_mask=ventricle_mask,
            ultrasound_image=ultrasound_resized
        )
        
        # Generate report
        print(analyzer.generate_report(measurement))
        
        # Visualize first result
        if idx == 1:
            brain_mask = analyzer.estimate_brain_region_from_ultrasound(ultrasound_resized)
            analyzer.visualize_analysis(
                ultrasound_resized,
                ventricle_mask,
                brain_mask,
                measurement,
                save_path='end_to_end_result.png'
            )
        
        results.append({
            'image': img_path.name,
            'vhr': measurement.vhr_percentage,
            'area_mm2': measurement.ventricle_area_mm2,
            'diameter_mm': measurement.equivalent_diameter_mm,
            'classification': measurement.severity_classification,
            'confidence': measurement.confidence
        })
    
    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nClassifications:")
    print(df['classification'].value_counts())
    print(f"\nAverage Metrics:")
    print(f"  Mean VHR: {df['vhr'].mean():.1f}%")
    print(f"  Mean Area: {df['area_mm2'].mean():.1f} mm²")
    print(f"  Mean Diameter: {df['diameter_mm'].mean():.1f} mm")
    print(f"  Mean Confidence: {df['confidence'].mean():.0%}")
    
    # Save results
    df.to_csv('end_to_end_results.csv', index=False)
    print(f"\nResults saved to end_to_end_results.csv")
    print("Visualization saved to end_to_end_result.png")
    
    return df


if __name__ == "__main__":
    results = run_end_to_end_analysis()
"""
VENTRICLE MEASUREMENT AND CLASSIFICATION PIPELINE
==================================================
Converts segmented masks into clinical measurements and diagnostic classifications.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VentricleMeasurement:
    """Data class to store ventricle measurements."""
    ventricle_area_pixels: int
    brain_area_pixels: int
    total_area_pixels: int
    vhr_percentage: float  # Ventricle-to-Hemisphere Ratio
    ventricle_area_mm2: float
    equivalent_diameter_mm: float
    severity_classification: str
    severity_score: int
    confidence: float


class VentricleAnalyzer:
    """
    Analyzes segmented masks to extract clinical measurements.
    Uses VHR (Ventricle-to-Hemisphere Ratio) for gestational-age-independent assessment.
    """
    
    # Clinical thresholds based on VHR
    VHR_THRESHOLDS = {
        'normal': 25.0,      # < 25% is normal
        'mild': 35.0,        # 25-35% is mild ventriculomegaly
        'moderate': 50.0,    # 35-50% is moderate
        # > 50% is severe (hydrocephalus)
    }
    
    # Approximate pixel spacing (mm per pixel)
    # This should ideally come from DICOM metadata
    DEFAULT_PIXEL_SPACING_MM = 0.5
    
    def __init__(self, pixel_spacing_mm: float = None):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        pixel_spacing_mm : float
            Physical size of each pixel in millimeters
        """
        self.pixel_spacing_mm = pixel_spacing_mm or self.DEFAULT_PIXEL_SPACING_MM
        logger.info(f"Initialized analyzer with pixel spacing: {self.pixel_spacing_mm} mm/pixel")
    
    def extract_brain_mask_from_original(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Extract brain tissue from the original multi-class segmentation.
        
        Parameters:
        -----------
        segmentation_mask : np.ndarray
            Original colored segmentation mask (H, W, 3)
            
        Returns:
        --------
        np.ndarray
            Binary mask of brain tissue
        """
        # Brain is red in BGR format (OpenCV): B=0, G=0, R=255
        if len(segmentation_mask.shape) == 3:
            brain_mask = (
                (segmentation_mask[:,:,0] == 0) &    # Blue channel
                (segmentation_mask[:,:,1] == 0) &    # Green channel
                (segmentation_mask[:,:,2] == 255)    # Red channel
            )
        else:
            # If grayscale, assume it's already processed
            brain_mask = segmentation_mask > 0
            
        return brain_mask.astype(np.uint8)
    
    def measure_ventricles(self, 
                          ventricle_mask: np.ndarray,
                          brain_mask: Optional[np.ndarray] = None,
                          original_image: Optional[np.ndarray] = None) -> VentricleMeasurement:
        """
        Extract comprehensive measurements from ventricle segmentation.
        
        Parameters:
        -----------
        ventricle_mask : np.ndarray
            Binary mask of ventricles (0 or 1/255)
        brain_mask : np.ndarray, optional
            Binary mask of brain tissue for VHR calculation
        original_image : np.ndarray, optional
            Original ultrasound for context
            
        Returns:
        --------
        VentricleMeasurement
            Complete measurement results
        """
        # Ensure binary mask
        ventricle_mask = (ventricle_mask > 0.5).astype(np.uint8)
        
        # Count pixels
        ventricle_pixels = np.sum(ventricle_mask)
        
        # If no brain mask provided, estimate from image
        if brain_mask is None:
            # Assume entire image minus black background is potential brain area
            if original_image is not None:
                brain_mask = (original_image > 20).astype(np.uint8)  # Threshold for non-background
            else:
                # Fallback: use entire image area
                brain_mask = np.ones_like(ventricle_mask)
        
        brain_pixels = np.sum(brain_mask)
        total_pixels = ventricle_mask.size
        
        # Calculate VHR (Ventricle-to-Hemisphere Ratio)
        if brain_pixels > 0:
            vhr = (ventricle_pixels / brain_pixels) * 100
        else:
            vhr = 0.0
        
        # Calculate physical measurements
        pixel_area_mm2 = self.pixel_spacing_mm ** 2
        ventricle_area_mm2 = ventricle_pixels * pixel_area_mm2
        
        # Equivalent diameter (diameter of circle with same area)
        if ventricle_area_mm2 > 0:
            equivalent_diameter_mm = 2 * np.sqrt(ventricle_area_mm2 / np.pi)
        else:
            equivalent_diameter_mm = 0.0
        
        # Classify severity based on VHR
        severity_classification, severity_score = self.classify_severity(vhr)
        
        # Calculate confidence based on mask quality
        confidence = self.calculate_confidence(ventricle_mask, brain_mask)
        
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
        """
        Classify severity based on VHR thresholds.
        
        Parameters:
        -----------
        vhr : float
            Ventricle-to-Hemisphere Ratio as percentage
            
        Returns:
        --------
        Tuple[str, int]
            (classification_name, severity_score)
        """
        if vhr < self.VHR_THRESHOLDS['normal']:
            return "Normal", 0
        elif vhr < self.VHR_THRESHOLDS['mild']:
            return "Mild Ventriculomegaly", 1
        elif vhr < self.VHR_THRESHOLDS['moderate']:
            return "Moderate Ventriculomegaly", 2
        else:
            return "Severe - Possible Hydrocephalus", 3
    
    def calculate_confidence(self, ventricle_mask: np.ndarray, brain_mask: np.ndarray) -> float:
        """
        Calculate confidence score based on mask characteristics.
        
        Parameters:
        -----------
        ventricle_mask : np.ndarray
            Binary ventricle mask
        brain_mask : np.ndarray
            Binary brain mask
            
        Returns:
        --------
        float
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Ventricles present
        if np.sum(ventricle_mask) > 100:  # At least 100 pixels
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Factor 2: Reasonable VHR
        vhr = (np.sum(ventricle_mask) / (np.sum(brain_mask) + 1)) * 100
        if 5 < vhr < 60:  # Physiologically plausible range
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.3)
        
        # Factor 3: Mask connectivity (not fragmented)
        num_components, _ = cv2.connectedComponents(ventricle_mask.astype(np.uint8))
        if num_components <= 3:  # Should be 1-2 ventricles
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)
    
    def generate_clinical_report(self, measurement: VentricleMeasurement) -> str:
        """
        Generate a clinical-style report from measurements.
        
        Parameters:
        -----------
        measurement : VentricleMeasurement
            Measurement results
            
        Returns:
        --------
        str
            Formatted clinical report
        """
        report = []
        report.append("FETAL VENTRICLE ANALYSIS REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Measurements
        report.append("MEASUREMENTS:")
        report.append(f"  Ventricle-to-Hemisphere Ratio: {measurement.vhr_percentage:.1f}%")
        report.append(f"  Ventricle Area: {measurement.ventricle_area_mm2:.1f} mm²")
        report.append(f"  Equivalent Diameter: {measurement.equivalent_diameter_mm:.1f} mm")
        report.append("")
        
        # Classification
        report.append("CLINICAL CLASSIFICATION:")
        report.append(f"  {measurement.severity_classification}")
        report.append(f"  Severity Score: {measurement.severity_score}/3")
        report.append(f"  Confidence: {measurement.confidence:.0%}")
        report.append("")
        
        # Clinical significance
        report.append("CLINICAL SIGNIFICANCE:")
        if measurement.severity_score == 0:
            report.append("  No intervention required. Normal ventricle size.")
        elif measurement.severity_score == 1:
            report.append("  Mild enlargement. Recommend follow-up ultrasound in 2-4 weeks.")
        elif measurement.severity_score == 2:
            report.append("  Moderate enlargement. Recommend detailed fetal MRI and specialist consultation.")
        else:
            report.append("  Severe enlargement suggestive of hydrocephalus.")
            report.append("  URGENT: Recommend immediate specialist referral and delivery planning.")
        
        report.append("")
        report.append("Note: This is an automated analysis. Clinical correlation required.")
        
        return "\n".join(report)


def analyze_test_set_measurements():
    """
    Analyze measurements from the test set predictions.
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.models.unet import UNet
    from src.data.ventricle_dataset_matched import VentricleDatasetMatched
    from torch.utils.data import DataLoader
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = VentricleAnalyzer()
    
    # Load model
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load('src/training/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = VentricleDatasetMatched(
        image_dir='data/raw/trans_ventricular_original',
        mask_dir='data/raw/trans_ventricular/segmentation_mask/SegmentationClass',
        csv_report='matching_report.csv',
        mode='test',
        img_size=256,
        transform=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Analyze each sample
    results = []
    
    print("\nAnalyzing test set measurements...")
    print("=" * 60)
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= 5:  # Analyze first 5 samples for demonstration
                break
            
            # Get data
            image = batch['image'].to(device)
            true_mask = batch['mask']
            
            # Predict
            prediction = model(image)
            pred_mask = torch.sigmoid(prediction) > 0.5
            pred_mask = pred_mask.cpu().numpy().squeeze()
            
            # Measure
            measurement = analyzer.measure_ventricles(
                ventricle_mask=pred_mask,
                brain_mask=None,  # Would need to load original colored mask
                original_image=image.cpu().numpy().squeeze()
            )
            
            # Generate report
            print(f"\nSample {idx + 1}:")
            print("-" * 40)
            print(analyzer.generate_clinical_report(measurement))
            
            results.append({
                'sample': idx,
                'vhr': measurement.vhr_percentage,
                'classification': measurement.severity_classification,
                'confidence': measurement.confidence
            })
    
    # Summary statistics
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("\nClassification Distribution:")
    print(df['classification'].value_counts())
    print(f"\nMean VHR: {df['vhr'].mean():.1f}%")
    print(f"Mean Confidence: {df['confidence'].mean():.0%}")
    
    return df


if __name__ == "__main__":
    # Run analysis on test set
    results_df = analyze_test_set_measurements()
    
    # Save results
    results_df.to_csv('measurement_results.csv', index=False)
    print(f"\nResults saved to measurement_results.csv")
"""
DUAL MODEL DIAGNOSTIC PIPELINE
==============================
Uses both ventricle and brain segmentation models for accurate VHR calculation.
This is the complete end-to-end system.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet import UNet


class CompleteDiagnosticPipeline:
    """
    Complete diagnostic system using dual models.
    - Ventricle model: Segments ventricles
    - Brain model: Segments brain tissue
    - Combines both for accurate VHR calculation
    """
    
    def __init__(self, 
                 ventricle_model_path: str = 'src/training/best_model.pth',
                 brain_model_path: str = 'brain_segmentation_model.pth',
                 pixel_spacing_mm: float = 0.5):
        """
        Initialize pipeline with both models.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pixel_spacing_mm = pixel_spacing_mm
        
        # Load ventricle model
        self.ventricle_model = UNet(in_channels=1, out_channels=1)
        checkpoint = torch.load(ventricle_model_path, map_location=self.device, weights_only=False)
        self.ventricle_model.load_state_dict(checkpoint['model_state_dict'])
        self.ventricle_model.to(self.device)
        self.ventricle_model.eval()
        print(f"✓ Loaded ventricle model")
        
        # Load brain model
        self.brain_model = UNet(in_channels=1, out_channels=1)
        checkpoint = torch.load(brain_model_path, map_location=self.device, weights_only=False)
        self.brain_model.load_state_dict(checkpoint['model_state_dict'])
        self.brain_model.to(self.device)
        self.brain_model.eval()
        print(f"✓ Loaded brain model")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Load and preprocess ultrasound image.
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_shape = image.shape
        
        # Resize to model input size
        image_resized = cv2.resize(image, (256, 256))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        return image_resized, image_tensor
    
    def segment_structures(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Segment both ventricles and brain using the two models.
        """
        with torch.no_grad():
            # Segment ventricles
            ventricle_pred = self.ventricle_model(image_tensor)
            ventricle_mask = torch.sigmoid(ventricle_pred) > 0.5
            ventricle_mask = ventricle_mask.cpu().numpy().squeeze()
            
            # Segment brain
            brain_pred = self.brain_model(image_tensor)
            brain_mask = torch.sigmoid(brain_pred) > 0.5
            brain_mask = brain_mask.cpu().numpy().squeeze()
        
        return {
            'ventricles': ventricle_mask.astype(np.uint8),
            'brain': brain_mask.astype(np.uint8)
        }
    
    def calculate_measurements(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate clinical measurements from segmentation masks.
        """
        ventricle_mask = masks['ventricles']
        brain_mask = masks['brain']
        
        # Count pixels
        ventricle_pixels = np.sum(ventricle_mask)
        brain_pixels = np.sum(brain_mask)
        
        # CORRECT VHR: Ventricle to HEMISPHERE (half brain) ratio
        hemisphere_pixels = brain_pixels / 2  # This is the key fix!
        
        if hemisphere_pixels > 0:
            vhr = (ventricle_pixels / hemisphere_pixels) * 100
        else:
            vhr = 0.0
        
        # Physical measurements
        pixel_area_mm2 = self.pixel_spacing_mm ** 2
        ventricle_area_mm2 = ventricle_pixels * pixel_area_mm2
        brain_area_mm2 = brain_pixels * pixel_area_mm2
        
        # Equivalent diameters
        if ventricle_area_mm2 > 0:
            ventricle_diameter_mm = 2 * np.sqrt(ventricle_area_mm2 / np.pi)
        else:
            ventricle_diameter_mm = 0.0
        
        if brain_area_mm2 > 0:
            brain_diameter_mm = 2 * np.sqrt(brain_area_mm2 / np.pi)
        else:
            brain_diameter_mm = 0.0
        
        return {
            'vhr_percentage': vhr,
            'ventricle_area_mm2': ventricle_area_mm2,
            'brain_area_mm2': brain_area_mm2,
            'ventricle_diameter_mm': ventricle_diameter_mm,
            'brain_diameter_mm': brain_diameter_mm,
            'ventricle_pixels': int(ventricle_pixels),
            'brain_pixels': int(brain_pixels)
        }
    
    def classify_severity(self, vhr: float) -> Tuple[str, int]:
        """
        Classify based on VHR thresholds.
        Note: These thresholds assume proper brain segmentation.
        """
        # Adjusted thresholds for expected VHR ranges
        if vhr < 10:
            return "Normal", 0
        elif vhr < 15:
            return "Mild Ventriculomegaly", 1
        elif vhr < 25:
            return "Moderate Ventriculomegaly", 2
        else:
            return "Severe - Possible Hydrocephalus", 3
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Complete analysis of a single ultrasound image.
        """
        # Preprocess
        image_array, image_tensor = self.preprocess_image(image_path)
        
        # Segment
        masks = self.segment_structures(image_tensor)
        
        # Measure
        measurements = self.calculate_measurements(masks)
        
        # Classify
        classification, severity_score = self.classify_severity(measurements['vhr_percentage'])
        
        # Compile results
        results = {
            'image_path': image_path,
            'masks': masks,
            'measurements': measurements,
            'classification': classification,
            'severity_score': severity_score,
            'image_array': image_array
        }
        
        return results
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(results['image_array'], cmap='gray')
        axes[0, 0].set_title('Original Ultrasound')
        axes[0, 0].axis('off')
        
        # Brain segmentation
        axes[0, 1].imshow(results['masks']['brain'], cmap='hot')
        axes[0, 1].set_title('Brain Segmentation\n(from brain model)')
        axes[0, 1].axis('off')
        
        # Ventricle segmentation
        axes[0, 2].imshow(results['masks']['ventricles'], cmap='hot')
        axes[0, 2].set_title('Ventricle Segmentation\n(from ventricle model)')
        axes[0, 2].axis('off')
        
        # Combined overlay
        overlay = results['image_array'].copy()
        overlay_rgb = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay_rgb[results['masks']['brain'] > 0] = [0, 100, 0]  # Green for brain
        overlay_rgb[results['masks']['ventricles'] > 0] = [255, 0, 0]  # Red for ventricles
        axes[1, 0].imshow(overlay_rgb)
        axes[1, 0].set_title('Combined Segmentation')
        axes[1, 0].axis('off')
        
        # Measurements text
        axes[1, 1].axis('off')
        measurements_text = f"""
        MEASUREMENTS:
        
        VHR: {results['measurements']['vhr_percentage']:.1f}%
        
        Ventricle Area: {results['measurements']['ventricle_area_mm2']:.1f} mm²
        Ventricle Diameter: {results['measurements']['ventricle_diameter_mm']:.1f} mm
        
        Brain Area: {results['measurements']['brain_area_mm2']:.1f} mm²
        Brain Diameter: {results['measurements']['brain_diameter_mm']:.1f} mm
        """
        axes[1, 1].text(0.1, 0.5, measurements_text, fontsize=12, verticalalignment='center')
        
        # Classification
        axes[1, 2].axis('off')
        severity_colors = ['green', 'yellow', 'orange', 'red']
        color = severity_colors[results['severity_score']]
        
        classification_text = f"""
        CLASSIFICATION:
        
        {results['classification']}
        
        Severity Score: {results['severity_score']}/3
        """
        axes[1, 2].text(0.1, 0.5, classification_text, fontsize=14, 
                       verticalalignment='center', color=color, weight='bold')
        
        plt.suptitle('Dual Model Diagnostic Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate clinical report.
        """
        report = []
        report.append("="*50)
        report.append("FETAL BRAIN VENTRICLE ANALYSIS REPORT")
        report.append("Dual Model Diagnostic System")
        report.append("="*50)
        report.append("")
        
        report.append("MEASUREMENTS:")
        report.append(f"  Ventricle-to-Brain Ratio (VHR): {results['measurements']['vhr_percentage']:.1f}%")
        report.append(f"  Ventricle Area: {results['measurements']['ventricle_area_mm2']:.1f} mm²")
        report.append(f"  Ventricle Diameter: {results['measurements']['ventricle_diameter_mm']:.1f} mm")
        report.append(f"  Brain Area: {results['measurements']['brain_area_mm2']:.1f} mm²")
        report.append("")
        
        report.append("CLASSIFICATION:")
        report.append(f"  {results['classification']}")
        report.append(f"  Severity Score: {results['severity_score']}/3")
        report.append("")
        
        report.append("CLINICAL RECOMMENDATION:")
        if results['severity_score'] == 0:
            report.append("  Normal ventricle size. Routine follow-up.")
        elif results['severity_score'] == 1:
            report.append("  Mild ventriculomegaly detected.")
            report.append("  Recommend: Follow-up ultrasound in 2-4 weeks.")
        elif results['severity_score'] == 2:
            report.append("  Moderate ventriculomegaly detected.")
            report.append("  Recommend: Detailed fetal MRI and specialist consultation.")
        else:
            report.append("  SEVERE ventriculomegaly/hydrocephalus suspected.")
            report.append("  URGENT: Immediate maternal-fetal medicine referral.")
            report.append("  Prepare for possible postnatal intervention.")
        
        report.append("")
        report.append("Note: This is an automated analysis. Clinical correlation required.")
        
        return "\n".join(report)


def test_complete_pipeline():
    """
    Test the complete dual-model pipeline on sample images.
    """
    print("\n" + "="*60)
    print("TESTING COMPLETE DUAL-MODEL DIAGNOSTIC PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CompleteDiagnosticPipeline()
    
    # Get test images
    image_dir = Path('data/raw/trans_ventricular_original')
    test_images = sorted(list(image_dir.glob('*.png')))[-5:]  # Last 5 images
    
    results_list = []
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"\n{'='*40}")
        print(f"Analyzing Image {idx}: {image_path.name}")
        print('='*40)
        
        # Analyze
        results = pipeline.analyze_image(str(image_path))
        
        # Print report
        print(pipeline.generate_report(results))
        
        # Visualize first one
        if idx == 1:
            pipeline.visualize_results(results, save_path='dual_model_results.png')
        
        # Store results
        results_list.append({
            'image': image_path.name,
            'vhr': results['measurements']['vhr_percentage'],
            'ventricle_area_mm2': results['measurements']['ventricle_area_mm2'],
            'ventricle_diameter_mm': results['measurements']['ventricle_diameter_mm'],
            'classification': results['classification'],
            'severity': results['severity_score']
        })
    
    # Summary statistics
    df = pd.DataFrame(results_list)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("\nClassification Distribution:")
    print(df['classification'].value_counts())
    print(f"\nAverage VHR: {df['vhr'].mean():.1f}%")
    print(f"VHR Range: {df['vhr'].min():.1f}% - {df['vhr'].max():.1f}%")
    
    # Save results
    df.to_csv('dual_model_analysis_results.csv', index=False)
    print("\nResults saved to: dual_model_analysis_results.csv")
    print("Visualization saved to: dual_model_results.png")


if __name__ == "__main__":
    test_complete_pipeline()
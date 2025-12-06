"""
COMPREHENSIVE MODEL EVALUATION SUITE
=====================================
Author: MD
Date: 2024
Purpose: Rigorous evaluation of fetal ventricle segmentation model
Following medical imaging evaluation standards and software engineering best practices
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation framework for medical image segmentation models.
    Follows MICCAI (Medical Image Computing) evaluation standards.
    """
    
    def __init__(self, 
                 model_path: Path,
                 data_dir: Path,
                 output_dir: Path = Path('evaluation_results'),
                 device: str = None):
        """
        Initialize the evaluator with proper configuration.
        
        Parameters:
        -----------
        model_path : Path
            Path to trained model checkpoint
        data_dir : Path
            Root directory containing data
        output_dir : Path
            Directory to save evaluation results
        device : str
            Computing device (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initialized evaluator on device: {self.device}")
        
        # Initialize metrics storage
        self.results = {
            'segmentation_metrics': {},
            'clinical_metrics': {},
            'per_sample_results': [],
            'statistical_analysis': {},
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'model_path': str(model_path),
                'device': str(self.device)
            }
        }
    
    def load_model(self):
        """
        Load the trained model with proper error handling.
        """
        try:
            # Import model architecture
            from src.models.unet import UNet
            
            # Initialize model
            self.model = UNet(in_channels=1, out_channels=1)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Store training metadata if available
            if 'epoch' in checkpoint:
                self.results['metadata']['training_epochs'] = checkpoint['epoch']
            if 'best_dice' in checkpoint:
                self.results['metadata']['training_best_dice'] = checkpoint['best_dice']
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def prepare_test_data(self):
        """
        Prepare test dataset with proper validation.
        """
        try:
            # Import dataset class
            from src.data.ventricle_dataset_matched import VentricleDatasetMatched
            
            # Set paths
            image_dir = self.data_dir / 'trans_ventricular_original'
            mask_dir = self.data_dir / 'trans_ventricular/segmentation_mask/SegmentationClass'
            csv_report = Path('matching_report.csv')
            
            # Verify paths exist
            assert image_dir.exists(), f"Image directory not found: {image_dir}"
            assert mask_dir.exists(), f"Mask directory not found: {mask_dir}"
            assert csv_report.exists(), f"CSV report not found: {csv_report}"
            
            # Create test dataset
            self.test_dataset = VentricleDatasetMatched(
                image_dir=image_dir,
                mask_dir=mask_dir,
                csv_report=csv_report,
                mode='test',
                img_size=256,
                transform=False  # No augmentation for testing
            )
            
            # Create dataloader
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=1,  # Process one at a time for detailed analysis
                shuffle=False,
                num_workers=0
            )
            
            logger.info(f"Test dataset prepared with {len(self.test_dataset)} samples")
            self.results['metadata']['test_samples'] = len(self.test_dataset)
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare test data: {str(e)}")
            return False
    
    def calculate_segmentation_metrics(self, 
                                      pred_mask: torch.Tensor, 
                                      true_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calculate comprehensive segmentation metrics.
        
        Metrics include:
        - Dice Similarity Coefficient (DSC)
        - Intersection over Union (IoU/Jaccard)
        - Sensitivity (Recall/True Positive Rate)
        - Specificity (True Negative Rate)
        - Precision (Positive Predictive Value)
        - Accuracy
        
        Parameters:
        -----------
        pred_mask : torch.Tensor
            Predicted segmentation mask
        true_mask : torch.Tensor
            Ground truth segmentation mask
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all metrics
        """
        # Ensure binary masks
        pred_mask = (pred_mask > 0.5).float()
        true_mask = (true_mask > 0.5).float()
        
        # Flatten for easier computation
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()
        
        # Calculate basic components
        true_positive = (pred_flat * true_flat).sum()
        false_positive = (pred_flat * (1 - true_flat)).sum()
        false_negative = ((1 - pred_flat) * true_flat).sum()
        true_negative = ((1 - pred_flat) * (1 - true_flat)).sum()
        
        # Small epsilon to avoid division by zero
        eps = 1e-7
        
        # Calculate metrics
        metrics = {}
        
        # Dice Similarity Coefficient
        metrics['dice'] = (2 * true_positive + eps) / (2 * true_positive + false_positive + false_negative + eps)
        
        # IoU/Jaccard Index
        metrics['iou'] = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
        
        # Sensitivity (Recall)
        metrics['sensitivity'] = (true_positive + eps) / (true_positive + false_negative + eps)
        
        # Specificity
        metrics['specificity'] = (true_negative + eps) / (true_negative + false_positive + eps)
        
        # Precision
        metrics['precision'] = (true_positive + eps) / (true_positive + false_positive + eps)
        
        # Accuracy
        metrics['accuracy'] = (true_positive + true_negative + eps) / (true_positive + true_negative + false_positive + false_negative + eps)
        
        # F1 Score (same as Dice for binary segmentation)
        metrics['f1'] = metrics['dice']
        
        # Convert to float for JSON serialization
        return {k: float(v) for k, v in metrics.items()}
    
    def calculate_clinical_metrics(self,
                                  pred_mask: np.ndarray,
                                  true_mask: np.ndarray,
                                  pixel_spacing: float = 0.5) -> Dict[str, float]:
        """
        Calculate clinically relevant metrics.
        
        Parameters:
        -----------
        pred_mask : np.ndarray
            Predicted mask
        true_mask : np.ndarray
            Ground truth mask
        pixel_spacing : float
            Physical size of each pixel in mm (default 0.5mm)
            
        Returns:
        --------
        Dict[str, float]
            Clinical measurements
        """
        metrics = {}
        
        # Calculate areas in pixels
        pred_area_pixels = np.sum(pred_mask > 0.5)
        true_area_pixels = np.sum(true_mask > 0.5)
        
        # Convert to mm²
        pixel_area_mm2 = pixel_spacing ** 2
        pred_area_mm2 = pred_area_pixels * pixel_area_mm2
        true_area_mm2 = true_area_pixels * pixel_area_mm2
        
        # Area difference
        area_diff_mm2 = abs(pred_area_mm2 - true_area_mm2)
        area_diff_percent = (area_diff_mm2 / (true_area_mm2 + 1e-7)) * 100
        
        # Store metrics
        metrics['predicted_area_mm2'] = float(pred_area_mm2)
        metrics['true_area_mm2'] = float(true_area_mm2)
        metrics['area_difference_mm2'] = float(area_diff_mm2)
        metrics['area_difference_percent'] = float(area_diff_percent)
        
        # Calculate equivalent diameter (diameter of circle with same area)
        pred_diameter_mm = 2 * np.sqrt(pred_area_mm2 / np.pi)
        true_diameter_mm = 2 * np.sqrt(true_area_mm2 / np.pi)
        
        metrics['predicted_diameter_mm'] = float(pred_diameter_mm)
        metrics['true_diameter_mm'] = float(true_diameter_mm)
        metrics['diameter_difference_mm'] = float(abs(pred_diameter_mm - true_diameter_mm))
        
        return metrics
    
    def run_evaluation(self):
        """
        Execute comprehensive model evaluation.
        """
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Prepare data
        if not self.prepare_test_data():
            return False
        
        # Initialize metric accumulators
        all_dice_scores = []
        all_iou_scores = []
        all_sensitivities = []
        all_specificities = []
        all_precisions = []
        all_area_differences = []
        
        # Process each test sample
        logger.info("Evaluating on test set...")
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader, desc='Evaluating')):
                # Get data
                images = batch['image'].to(self.device)
                true_masks = batch['mask'].to(self.device)
                vhr = batch['vhr'].item()
                severity = batch['severity'].item()
                img_name = batch['img_name'][0]
                
                # Predict
                predictions = self.model(images)
                pred_masks = torch.sigmoid(predictions)
                
                # Calculate segmentation metrics
                seg_metrics = self.calculate_segmentation_metrics(pred_masks, true_masks)
                
                # Calculate clinical metrics
                pred_np = pred_masks.cpu().numpy().squeeze()
                true_np = true_masks.cpu().numpy().squeeze()
                clin_metrics = self.calculate_clinical_metrics(pred_np, true_np)
                
                # Store results
                sample_result = {
                    'sample_id': idx,
                    'image_name': img_name,
                    'true_vhr': vhr,
                    'true_severity': severity,
                    **seg_metrics,
                    **clin_metrics
                }
                self.results['per_sample_results'].append(sample_result)
                
                # Accumulate for statistics
                all_dice_scores.append(seg_metrics['dice'])
                all_iou_scores.append(seg_metrics['iou'])
                all_sensitivities.append(seg_metrics['sensitivity'])
                all_specificities.append(seg_metrics['specificity'])
                all_precisions.append(seg_metrics['precision'])
                all_area_differences.append(clin_metrics['area_difference_percent'])
        
        # Calculate summary statistics
        self.calculate_summary_statistics(
            all_dice_scores,
            all_iou_scores,
            all_sensitivities,
            all_specificities,
            all_precisions,
            all_area_differences
        )
        
        # Generate report
        self.generate_report()
        
        # Create visualizations
        self.create_visualizations()
        
        logger.info("Evaluation complete!")
        return True
    
    def calculate_summary_statistics(self, dice_scores, iou_scores, sensitivities, 
                                    specificities, precisions, area_differences):
        """
        Calculate summary statistics with confidence intervals.
        """
        import scipy.stats as stats
        
        def calculate_stats(data, name):
            """Calculate mean, std, and 95% CI"""
            data = np.array(data)
            mean = np.mean(data)
            std = np.std(data)
            sem = stats.sem(data)  # Standard error of mean
            ci_95 = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            
            return {
                f'{name}_mean': float(mean),
                f'{name}_std': float(std),
                f'{name}_min': float(np.min(data)),
                f'{name}_max': float(np.max(data)),
                f'{name}_median': float(np.median(data)),
                f'{name}_q1': float(np.percentile(data, 25)),
                f'{name}_q3': float(np.percentile(data, 75)),
                f'{name}_ci_95_lower': float(ci_95[0]),
                f'{name}_ci_95_upper': float(ci_95[1])
            }
        
        # Calculate statistics for each metric
        self.results['segmentation_metrics'] = {
            **calculate_stats(dice_scores, 'dice'),
            **calculate_stats(iou_scores, 'iou'),
            **calculate_stats(sensitivities, 'sensitivity'),
            **calculate_stats(specificities, 'specificity'),
            **calculate_stats(precisions, 'precision')
        }
        
        self.results['clinical_metrics'] = calculate_stats(area_differences, 'area_difference_percent')
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        logger.info(f"Dice Score: {self.results['segmentation_metrics']['dice_mean']:.4f} ± {self.results['segmentation_metrics']['dice_std']:.4f}")
        logger.info(f"IoU: {self.results['segmentation_metrics']['iou_mean']:.4f} ± {self.results['segmentation_metrics']['iou_std']:.4f}")
        logger.info(f"Sensitivity: {self.results['segmentation_metrics']['sensitivity_mean']:.4f}")
        logger.info(f"Specificity: {self.results['segmentation_metrics']['specificity_mean']:.4f}")
        logger.info(f"Precision: {self.results['segmentation_metrics']['precision_mean']:.4f}")
    
    def generate_report(self):
        """
        Generate comprehensive evaluation report.
        """
        # Save JSON report
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")
        
        # Generate markdown report
        self.generate_markdown_report()
        
        # Generate CSV for per-sample results
        df = pd.DataFrame(self.results['per_sample_results'])
        csv_path = self.output_dir / 'per_sample_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Per-sample results saved to {csv_path}")
    
    def generate_markdown_report(self):
        """
        Generate human-readable markdown report.
        """
        report = []
        report.append("# Fetal Ventricle Segmentation Model Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model information
        report.append("\n## Model Information")
        report.append(f"- Model Path: `{self.model_path}`")
        report.append(f"- Device: {self.device}")
        report.append(f"- Test Samples: {self.results['metadata']['test_samples']}")
        
        # Segmentation metrics
        report.append("\n## Segmentation Performance Metrics")
        
        metrics_table = []
        metrics_table.append("| Metric | Mean ± Std | Median | Min-Max | 95% CI |")
        metrics_table.append("|--------|------------|--------|---------|--------|")
        
        for metric in ['dice', 'iou', 'sensitivity', 'specificity', 'precision']:
            mean = self.results['segmentation_metrics'][f'{metric}_mean']
            std = self.results['segmentation_metrics'][f'{metric}_std']
            median = self.results['segmentation_metrics'][f'{metric}_median']
            min_val = self.results['segmentation_metrics'][f'{metric}_min']
            max_val = self.results['segmentation_metrics'][f'{metric}_max']
            ci_lower = self.results['segmentation_metrics'][f'{metric}_ci_95_lower']
            ci_upper = self.results['segmentation_metrics'][f'{metric}_ci_95_upper']
            
            metrics_table.append(
                f"| {metric.capitalize()} | {mean:.4f} ± {std:.4f} | {median:.4f} | "
                f"{min_val:.4f}-{max_val:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |"
            )
        
        report.extend(metrics_table)
        
        # Clinical metrics
        report.append("\n## Clinical Measurement Accuracy")
        mean_area_diff = self.results['clinical_metrics']['area_difference_percent_mean']
        std_area_diff = self.results['clinical_metrics']['area_difference_percent_std']
        report.append(f"- Mean Area Difference: {mean_area_diff:.2f}% ± {std_area_diff:.2f}%")
        
        # Clinical interpretation
        report.append("\n## Clinical Interpretation")
        dice_mean = self.results['segmentation_metrics']['dice_mean']
        
        if dice_mean >= 0.80:
            interpretation = "Excellent - Suitable for clinical deployment with supervision"
        elif dice_mean >= 0.70:
            interpretation = "Good - Suitable for clinical research and validation studies"
        elif dice_mean >= 0.60:
            interpretation = "Moderate - Requires improvement before clinical use"
        else:
            interpretation = "Poor - Significant improvements needed"
        
        report.append(f"- Performance Level: **{interpretation}**")
        report.append(f"- Dice Score: {dice_mean:.4f}")
        
        # Recommendations
        report.append("\n## Recommendations")
        if dice_mean < 0.80:
            report.append("- Consider additional training with more diverse data")
            report.append("- Review cases with lowest Dice scores for failure patterns")
        if self.results['segmentation_metrics']['sensitivity_mean'] < 0.80:
            report.append("- Low sensitivity suggests model missing some ventricle regions")
        if self.results['segmentation_metrics']['specificity_mean'] < 0.95:
            report.append("- Consider adjusting threshold to reduce false positives")
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        logger.info(f"Markdown report saved to {report_path}")
    
    def create_visualizations(self):
        """
        Create comprehensive visualization plots.
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Dice score distribution
        ax1 = plt.subplot(2, 3, 1)
        dice_scores = [r['dice'] for r in self.results['per_sample_results']]
        ax1.hist(dice_scores, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(dice_scores), color='red', linestyle='--', label=f'Mean: {np.mean(dice_scores):.3f}')
        ax1.set_xlabel('Dice Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Dice Score Distribution')
        ax1.legend()
        
        # 2. IoU distribution
        ax2 = plt.subplot(2, 3, 2)
        iou_scores = [r['iou'] for r in self.results['per_sample_results']]
        ax2.hist(iou_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.3f}')
        ax2.set_xlabel('IoU Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('IoU Distribution')
        ax2.legend()
        
        # 3. Box plot of all metrics
        ax3 = plt.subplot(2, 3, 3)
        metrics_data = [
            [r['dice'] for r in self.results['per_sample_results']],
            [r['iou'] for r in self.results['per_sample_results']],
            [r['sensitivity'] for r in self.results['per_sample_results']],
            [r['specificity'] for r in self.results['per_sample_results']],
            [r['precision'] for r in self.results['per_sample_results']]
        ]
        ax3.boxplot(metrics_data, labels=['Dice', 'IoU', 'Sens', 'Spec', 'Prec'])
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics Comparison')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot: True vs Predicted Area
        ax4 = plt.subplot(2, 3, 4)
        true_areas = [r['true_area_mm2'] for r in self.results['per_sample_results']]
        pred_areas = [r['predicted_area_mm2'] for r in self.results['per_sample_results']]
        ax4.scatter(true_areas, pred_areas, alpha=0.6)
        ax4.plot([0, max(true_areas)], [0, max(true_areas)], 'r--', label='Perfect prediction')
        ax4.set_xlabel('True Area (mm²)')
        ax4.set_ylabel('Predicted Area (mm²)')
        ax4.set_title('Area Prediction Accuracy')
        ax4.legend()
        
        # 5. Performance by severity
        ax5 = plt.subplot(2, 3, 5)
        severity_names = ['Normal', 'Mild', 'Moderate', 'Severe']
        severity_dice = {}
        for r in self.results['per_sample_results']:
            sev = r['true_severity']
            if sev not in severity_dice:
                severity_dice[sev] = []
            severity_dice[sev].append(r['dice'])
        
        if severity_dice:
            positions = []
            dice_by_severity = []
            labels = []
            for sev in sorted(severity_dice.keys()):
                if sev < len(severity_names):
                    positions.append(sev)
                    dice_by_severity.append(severity_dice[sev])
                    labels.append(f"{severity_names[sev]}\n(n={len(severity_dice[sev])})")
            
            if dice_by_severity:
                bp = ax5.boxplot(dice_by_severity, positions=positions, labels=labels)
                ax5.set_ylabel('Dice Score')
                ax5.set_title('Performance by Severity')
                ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative distribution
        ax6 = plt.subplot(2, 3, 6)
        sorted_dice = np.sort(dice_scores)
        cumulative = np.arange(1, len(sorted_dice) + 1) / len(sorted_dice)
        ax6.plot(sorted_dice, cumulative, linewidth=2)
        ax6.axvline(0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
        ax6.axvline(0.8, color='gold', linestyle='--', alpha=0.7, label='Excellent (0.8)')
        ax6.set_xlabel('Dice Score')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('Cumulative Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Model Evaluation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plot_path = self.output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Evaluation plots saved to {plot_path}")
        
        # Create confusion matrix for severity classification if applicable
        self.create_severity_analysis()
    
    def create_severity_analysis(self):
        """
        Analyze performance based on clinical severity.
        """
        # This would require prediction of severity from the segmented area
        # For now, we'll analyze based on true severity
        
        df = pd.DataFrame(self.results['per_sample_results'])
        
        if 'true_severity' in df.columns:
            severity_stats = df.groupby('true_severity').agg({
                'dice': ['mean', 'std', 'min', 'max', 'count'],
                'iou': ['mean', 'std'],
                'area_difference_percent': ['mean', 'std']
            })
            
            # Save severity analysis
            severity_path = self.output_dir / 'severity_analysis.csv'
            severity_stats.to_csv(severity_path)
            logger.info(f"Severity analysis saved to {severity_path}")


def main():
    """
    Main execution function.
    """
    # Configuration
    MODEL_PATH = Path('src/training/best_model.pth')
    DATA_DIR = Path('data/raw')
    OUTPUT_DIR = Path('evaluation_results')
    
    # Verify paths
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return
    
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found at {DATA_DIR}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Run evaluation
    success = evaluator.run_evaluation()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info("="*60)
    else:
        logger.error("Evaluation failed. Check logs for details.")


if __name__ == "__main__":
    main()
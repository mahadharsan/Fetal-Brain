"""
Validate Grad-CAM Explainability
=================================
Generate Grad-CAM visualizations for diverse cases to verify
the model is focusing on correct anatomical regions.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.dual_model_pipeline_LVW_gradcam_confidence import CompleteDiagnosticPipeline

# Create output folder
OUTPUT_DIR = Path('gradcam_validation_results')
OUTPUT_DIR.mkdir(exist_ok=True)

def validate_gradcam_cases():
    """
    Test Grad-CAM on diverse cases:
    - Small ventricles (normal)
    - Medium ventricles (borderline)
    - Large ventricles (abnormal)
    """
    
    print("="*60)
    print("GRAD-CAM EXPLAINABILITY VALIDATION")
    print("="*60)
    print(f"\nOutput folder: {OUTPUT_DIR}/")
    
    # Initialize pipeline
    pipeline = CompleteDiagnosticPipeline(
        ventricle_model_path='src/training/best_model.pth',
        brain_model_path='brain_segmentation_model.pth',
        pixel_size_csv='data/raw/trans_ventricular/Trans-ventricular-Pixel-Size.csv'
    )
    
    # Test cases - UPDATED with actual files
    test_cases = {
        'Small Normal': 'Patient00832_Plane3_2_of_3',
        'Medium Borderline': 'Patient01301_Plane3_3_of_7',
        'Large Mild VM ': 'Patient01123_Plane3_4_of_4',
    }
    
    print("\n1. Analyzing test cases...")
    
    image_dir = Path('data/raw/trans_ventricular_original')
    results_all = {}
    
    for case_name, filename in test_cases.items():
        image_path = image_dir / f"{filename}.png"
        
        if not image_path.exists():
            print(f"\n⚠️ Image not found: {image_path}")
            continue
        
        print(f"\n📊 Processing: {case_name}")
        print(f"   File: {filename}")
        
        # Analyze with Grad-CAM
        results = pipeline.analyze_image(str(image_path))
        
        print(f"   Width: {results['measurements']['lateral_ventricle_width_mm']:.2f}mm")
        print(f"   Classification: {results['classification']}")
        print(f"   Confidence: {results['confidence']['overall_confidence']:.1f}%")
        
        results_all[case_name] = results
    
    # Create comprehensive visualization
    print("\n2. Creating Grad-CAM validation visualizations...")
    create_gradcam_validation_figure(results_all)
    
    # Save detailed analysis
    save_validation_report(results_all)
    
    print("\n" + "="*60)
    print("✅ GRAD-CAM VALIDATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    print("  - gradcam_validation.png - Visual validation")
    print("  - validation_report.txt - Detailed metrics")
    print("\nManual check required:")
    print("  ✓ Do heatmaps highlight ventricle regions?")
    print("  ✓ Are irrelevant areas ignored?")
    print("  ✓ Is attention stronger for larger ventricles?")

def create_gradcam_validation_figure(results_dict):
    """
    Create comparison figure showing Grad-CAM for different cases.
    """
    n_cases = len(results_dict)
    
    if n_cases == 0:
        print("⚠️ No results to visualize")
        return
    
    fig, axes = plt.subplots(n_cases, 5, figsize=(20, 4*n_cases))
    
    # Handle single case
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (case_name, results) in enumerate(results_dict.items()):
        
        # Column 1: Original image
        axes[idx, 0].imshow(results['image_array'], cmap='gray')
        axes[idx, 0].set_title(f"{case_name}\nOriginal Image", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Column 2: Ventricle mask (ground truth segmentation)
        axes[idx, 1].imshow(results['masks']['ventricles'], cmap='hot')
        axes[idx, 1].set_title(f"Ventricle Segmentation\nWidth: {results['measurements']['lateral_ventricle_width_mm']:.2f}mm", 
                              fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Column 3: Grad-CAM heatmap
        axes[idx, 2].imshow(results['heatmap'], cmap='jet')
        axes[idx, 2].set_title(f"Grad-CAM Attention\nConf: {results['confidence']['overall_confidence']:.1f}%", 
                              fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Column 4: Overlay
        axes[idx, 3].imshow(results['heatmap_overlay'])
        axes[idx, 3].set_title("Attention Overlay\n(Where model looks)", fontsize=10, fontweight='bold')
        axes[idx, 3].axis('off')
        
        # Column 5: Validation check (overlap visualization)
        axes[idx, 4].axis('off')
        
        # Create overlap visualization
        overlap_viz = np.zeros((256, 256, 3))
        
        # Red channel: ventricle mask (where ventricles ARE)
        overlap_viz[:, :, 0] = results['masks']['ventricles']
        
        # Green channel: high attention areas (where model LOOKS)
        high_attention = results['heatmap'] > 0.5
        overlap_viz[:, :, 1] = high_attention
        
        # Yellow = overlap (red + green) = CORRECT attention
        axes[idx, 4].imshow(overlap_viz)
        axes[idx, 4].set_title("Validation Check\n(Yellow=Correct)", fontsize=10, fontweight='bold')
        axes[idx, 4].axis('off')
        
        # Calculate overlap percentage
        overlap_pixels = np.sum(results['masks']['ventricles'] * high_attention)
        total_ventricle = np.sum(results['masks']['ventricles'])
        total_attention = np.sum(high_attention)
        
        if total_ventricle > 0:
            overlap_pct = (overlap_pixels / total_ventricle) * 100
        else:
            overlap_pct = 0
        
        # Determine if validation passed
        if overlap_pct > 60:
            status = "✓ PASS"
            color = 'green'
        elif overlap_pct > 40:
            status = "~ MARGINAL"
            color = 'orange'
        else:
            status = "✗ FAIL"
            color = 'red'
            
        axes[idx, 4].text(0.5, -0.15, f"Overlap: {overlap_pct:.1f}%\n{status}", 
                         transform=axes[idx, 4].transAxes,
                         ha='center', fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.suptitle('Grad-CAM Explainability Validation\nVerifying Model Attention Aligns with Ventricle Regions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'gradcam_validation.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    plt.show()

def save_validation_report(results_dict):
    """Save detailed validation metrics to text file"""
    
    report_file = OUTPUT_DIR / 'validation_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("GRAD-CAM VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        for case_name, results in results_dict.items():
            f.write(f"\n{case_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Filename: {Path(results['image_path']).name}\n")
            f.write(f"Ventricle Width: {results['measurements']['lateral_ventricle_width_mm']:.2f}mm\n")
            f.write(f"Classification: {results['classification']}\n")
            f.write(f"Confidence: {results['confidence']['overall_confidence']:.1f}%\n\n")
            
            # Calculate attention metrics
            ventricle_mask = results['masks']['ventricles']
            high_attention = results['heatmap'] > 0.5
            
            overlap_pixels = np.sum(ventricle_mask * high_attention)
            total_ventricle = np.sum(ventricle_mask)
            total_attention = np.sum(high_attention)
            
            if total_ventricle > 0:
                overlap_pct = (overlap_pixels / total_ventricle) * 100
                precision = (overlap_pixels / total_attention * 100) if total_attention > 0 else 0
            else:
                overlap_pct = 0
                precision = 0
            
            f.write(f"Attention Metrics:\n")
            f.write(f"  - Ventricle coverage: {overlap_pct:.1f}% (% of ventricle with attention)\n")
            f.write(f"  - Attention precision: {precision:.1f}% (% of attention on ventricle)\n")
            f.write(f"  - Ventricle pixels: {int(total_ventricle)}\n")
            f.write(f"  - High attention pixels: {int(total_attention)}\n")
            f.write(f"  - Overlap pixels: {int(overlap_pixels)}\n\n")
            
            # Validation status
            if overlap_pct > 60:
                f.write("  ✓ VALIDATION: PASS - Good attention alignment\n")
            elif overlap_pct > 40:
                f.write("  ~ VALIDATION: MARGINAL - Some misalignment\n")
            else:
                f.write("  ✗ VALIDATION: FAIL - Poor attention alignment\n")
            
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("SUMMARY\n")
        f.write("="*60 + "\n")
        f.write("\nGrad-CAM should show:\n")
        f.write("  1. High attention (red/yellow) over ventricle regions\n")
        f.write("  2. Low attention on brain edges and artifacts\n")
        f.write("  3. Stronger attention for larger ventricles\n\n")
        f.write("Expected overlap: >60% for good explainability\n")
    
    print(f"   Saved: {report_file}")

if __name__ == "__main__":
    validate_gradcam_cases()
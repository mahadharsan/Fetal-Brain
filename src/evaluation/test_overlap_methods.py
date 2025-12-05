"""
Test different overlap calculation methods
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.evaluation.dual_model_pipeline_LVW_gradcam_confidence import CompleteDiagnosticPipeline

def calculate_weighted_overlap(heatmap, ventricle_mask):
    """Continuous weighted overlap"""
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    mask_norm = ventricle_mask.astype(float)
    
    overlap_score = np.sum(heatmap_norm * mask_norm)
    max_possible = np.sum(mask_norm)
    
    if max_possible > 0:
        overlap_pct = (overlap_score / max_possible) * 100
    else:
        overlap_pct = 0
    
    return overlap_pct

def calculate_iou_overlap(heatmap, ventricle_mask, threshold=0.3):
    """IoU with lower threshold"""
    attention_region = (heatmap > threshold).astype(float)
    
    intersection = np.sum(attention_region * ventricle_mask)
    union = np.sum(attention_region) + np.sum(ventricle_mask) - intersection
    
    if union > 0:
        iou = (intersection / union) * 100
    else:
        iou = 0
    
    return iou

def calculate_center_alignment(heatmap, ventricle_mask):
    """Center-of-mass alignment"""
    from scipy import ndimage
    
    if np.sum(ventricle_mask) == 0:
        return 0
    
    ventricle_center = ndimage.center_of_mass(ventricle_mask)
    
    if np.sum(heatmap) == 0:
        return 0
        
    attention_center = ndimage.center_of_mass(heatmap)
    
    distance = np.sqrt((ventricle_center[0] - attention_center[0])**2 + 
                      (ventricle_center[1] - attention_center[1])**2)
    
    max_distance = np.sqrt(256**2 + 256**2)
    alignment_score = (1 - distance / max_distance) * 100
    
    return alignment_score

def test_overlap_methods():
    """Compare different overlap calculation methods"""
    
    print("="*60)
    print("TESTING DIFFERENT OVERLAP METHODS")
    print("="*60)
    
    # Initialize pipeline
    pipeline = CompleteDiagnosticPipeline(
        ventricle_model_path='src/training/best_model.pth',
        brain_model_path='brain_segmentation_model.pth',
        pixel_size_csv='data/raw/trans_ventricular/Trans-ventricular-Pixel-Size.csv'
    )
    
    # Test cases
    test_cases = {
        'Small (4.8mm)': 'Patient01652_Plane3_1_of_2',
        'Medium (8.1mm)': 'Patient01301_Plane3_3_of_7',
        'Large (11.6mm)': 'Patient01710_Plane3_4_of_5',
    }
    
    results = []
    
    image_dir = Path('data/raw/trans_ventricular_original')
    
    for case_name, filename in test_cases.items():
        image_path = image_dir / f"{filename}.png"
        
        print(f"\n{case_name}: {filename}")
        
        # Analyze
        analysis = pipeline.analyze_image(str(image_path))
        
        heatmap = analysis['heatmap']
        ventricle_mask = analysis['masks']['ventricles']
        
        # Method 1: Current (binary threshold 0.5)
        binary_overlap = np.sum((heatmap > 0.5) * ventricle_mask) / np.sum(ventricle_mask) * 100 if np.sum(ventricle_mask) > 0 else 0
        
        # Method 2: Weighted
        weighted = calculate_weighted_overlap(heatmap, ventricle_mask)
        
        # Method 3: IoU (threshold 0.3)
        iou = calculate_iou_overlap(heatmap, ventricle_mask, threshold=0.3)
        
        # Method 4: Center alignment
        center = calculate_center_alignment(heatmap, ventricle_mask)
        
        print(f"  Current method (binary 0.5): {binary_overlap:.1f}%")
        print(f"  Weighted overlap:            {weighted:.1f}%")
        print(f"  IoU (threshold 0.3):         {iou:.1f}%")
        print(f"  Center alignment:            {center:.1f}%")
        
        results.append({
            'case': case_name,
            'binary_0.5': binary_overlap,
            'weighted': weighted,
            'iou_0.3': iou,
            'center': center
        })
    
    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\nWhich method gives most reasonable scores?")
    print("- Binary 0.5: Strictest, may underestimate")
    print("- Weighted: Accounts for attention strength")
    print("- IoU 0.3: More lenient threshold")
    print("- Center: Focuses on alignment, not coverage")

if __name__ == "__main__":
    test_overlap_methods()
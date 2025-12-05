"""
COMPREHENSIVE TEST SET EVALUATION
==================================
Evaluates the complete pipeline on all 89 test images.
This gives us comprehensive performance metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import your latest pipeline
from src.evaluation.dual_model_pipeline_LVW_gradcam_confidence import CompleteDiagnosticPipeline

def load_test_set():
    """Load test image filenames from test_set_images.txt"""
    test_file = 'test_set_images.txt'
    with open(test_file, 'r') as f:
        test_filenames = [line.strip() for line in f.readlines()]
    return test_filenames

def evaluate_test_set():
    """
    Run complete evaluation on all test images.
    """
    print("="*60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)
    
    # Load test set
    print("\n1. Loading test set...")
    test_filenames = load_test_set()
    print(f"   Found {len(test_filenames)} test images")
    
    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = CompleteDiagnosticPipeline(
        ventricle_model_path='src/training/best_model.pth',
        brain_model_path='brain_segmentation_model.pth',
        pixel_size_csv='data/raw/trans_ventricular/Trans-ventricular-Pixel-Size.csv'
    )
    
    # Process all test images
    print("\n3. Evaluating all test images...")
    image_dir = Path('data/raw/trans_ventricular_original')
    
    results_list = []
    failed_images = []
    zero_ventricle_images = []  # NEW: Track images with 0mm ventricles
    
    for filename in tqdm(test_filenames, desc="Processing"):
        image_path = image_dir / f"{filename}.png"
        
        # Check if image exists
        if not image_path.exists():
            print(f"⚠️ WARNING: Image not found: {image_path}")
            failed_images.append(filename)
            continue
        
        try:
            # Analyze image
            results = pipeline.analyze_image(str(image_path))
            
            ventricle_width = results['measurements']['lateral_ventricle_width_mm']
            
            # NEW: Check for zero ventricle width
            if ventricle_width == 0.0 or ventricle_width < 0.1:
                zero_ventricle_images.append({
                    'filename': filename,
                    'ventricle_width': ventricle_width,
                    'confidence': results['confidence']['overall_confidence'],
                    'ventricle_pixels': results['measurements']['ventricle_pixels'],
                    'reason': 'No ventricles detected' if ventricle_width == 0.0 else 'Very small detection'
                })
                print(f"\n⚠️ Zero/near-zero ventricle width: {filename} ({ventricle_width:.2f}mm)")
                # Skip adding to main results - don't include in statistics
                continue
            
            # Extract key metrics (only for valid detections)
            results_list.append({
                'filename': filename,
                'lateral_ventricle_width_mm': ventricle_width,
                'vhr_percentage': results['measurements']['vhr_percentage'],
                'ventricle_area_mm2': results['measurements']['ventricle_area_mm2'],
                'brain_area_mm2': results['measurements']['brain_area_mm2'],
                'ventricle_diameter_mm': results['measurements']['ventricle_diameter_mm'],
                'brain_diameter_mm': results['measurements']['brain_diameter_mm'],
                'classification': results['classification'],
                'severity_score': results['severity_score'],
                'confidence': results['confidence']['overall_confidence'],
                'high_confidence': results['confidence']['high_confidence'],
                'low_confidence_warning': results['confidence']['warning']
            })
            
        except Exception as e:
            print(f"⚠️ ERROR processing {filename}: {str(e)}")
            failed_images.append(filename)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Save raw results (valid detections only)
    output_csv = 'test_set_comprehensive_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    # NEW: Save zero ventricle cases to separate file
    if zero_ventricle_images:
        zero_df = pd.DataFrame(zero_ventricle_images)
        zero_output = 'zero_ventricle_cases.csv'
        zero_df.to_csv(zero_output, index=False)
        print(f"⚠️ Zero ventricle cases saved to: {zero_output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TEST SET EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total test images: {len(test_filenames)}")
    print(f"   Successfully processed (valid detections): {len(results_list)}")
    print(f"   Zero/near-zero ventricle width: {len(zero_ventricle_images)}")
    print(f"   Failed to process: {len(failed_images)}")
    
    if zero_ventricle_images:
        print(f"\n⚠️ Images with zero/near-zero ventricle width:")
        for case in zero_ventricle_images[:10]:  # Show first 10
            print(f"      - {case['filename']}: {case['ventricle_width']:.2f}mm (confidence: {case['confidence']:.1f}%, {case['reason']})")
        if len(zero_ventricle_images) > 10:
            print(f"      ... and {len(zero_ventricle_images) - 10} more (see zero_ventricle_cases.csv)")
    
    if failed_images:
        print(f"\n⚠️ Failed images:")
        for img in failed_images[:10]:  # Show first 10
            print(f"      - {img}")
    
    # Only calculate statistics if we have valid results
    if len(df) > 0:
        print(f"\n📏 Lateral Ventricle Width (Primary Metric) - Valid Detections Only:")
        print(f"   N = {len(df)} images")
        print(f"   Mean: {df['lateral_ventricle_width_mm'].mean():.2f} mm")
        print(f"   Std: {df['lateral_ventricle_width_mm'].std():.2f} mm")
        print(f"   Min: {df['lateral_ventricle_width_mm'].min():.2f} mm")
        print(f"   Max: {df['lateral_ventricle_width_mm'].max():.2f} mm")
        print(f"   Median: {df['lateral_ventricle_width_mm'].median():.2f} mm")
        
        print(f"\n📊 VHR (Reference only):")
        print(f"   Mean: {df['vhr_percentage'].mean():.2f}%")
        print(f"   Std: {df['vhr_percentage'].std():.2f}%")
        
        print(f"\n🏥 Classification Distribution:")
        class_counts = df['classification'].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Confidence Analysis:")
        print(f"   Mean confidence: {df['confidence'].mean():.1f}%")
        print(f"   High confidence cases (>75%): {df['high_confidence'].sum()} ({df['high_confidence'].sum()/len(df)*100:.1f}%)")
        print(f"   Low confidence cases (<50%): {df['low_confidence_warning'].sum()} ({df['low_confidence_warning'].sum()/len(df)*100:.1f}%)")
        
        # Generate visualizations
        print("\n4. Generating visualizations...")
        create_visualizations(df, len(zero_ventricle_images))
    else:
        print("\n⚠️ No valid detections to analyze!")
    
    return df, zero_ventricle_images

def create_visualizations(df, num_zero_cases):
    """
    Create comprehensive visualizations of test set results.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Distribution of Lateral Ventricle Width
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df['lateral_ventricle_width_mm'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10mm threshold')
    ax1.set_xlabel('Lateral Ventricle Width (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Ventricle Width\n(Excluding {num_zero_cases} zero cases)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Classification pie chart
    ax2 = plt.subplot(3, 3, 2)
    class_counts = df['classification'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors[:len(class_counts)], startangle=90)
    ax2.set_title('Classification Distribution')
    
    # 3. Confidence distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['confidence'], bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax3.axvline(x=75, color='green', linestyle='--', label='High confidence')
    ax3.axvline(x=50, color='red', linestyle='--', label='Low confidence')
    ax3.set_xlabel('Confidence Score (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter: Width vs Confidence
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(df['lateral_ventricle_width_mm'], df['confidence'], 
                         c=df['severity_score'], cmap='RdYlGn_r', alpha=0.6)
    ax4.set_xlabel('Lateral Ventricle Width (mm)')
    ax4.set_ylabel('Confidence Score (%)')
    ax4.set_title('Width vs Confidence')
    ax4.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax4, label='Severity Score')
    ax4.grid(True, alpha=0.3)
    
    # 5. VHR distribution (for reference)
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(df['vhr_percentage'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax5.set_xlabel('VHR (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('VHR Distribution (Reference)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot of width by classification
    ax6 = plt.subplot(3, 3, 6)
    classifications = df['classification'].unique()
    data_by_class = [df[df['classification'] == c]['lateral_ventricle_width_mm'].values 
                     for c in classifications]
    ax6.boxplot(data_by_class, labels=classifications)
    ax6.set_ylabel('Lateral Ventricle Width (mm)')
    ax6.set_title('Width Distribution by Classification')
    ax6.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # 7. Confidence by severity
    ax7 = plt.subplot(3, 3, 7)
    severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe']
    conf_by_severity = [df[df['severity_score'] == i]['confidence'].values 
                       for i in range(4) if len(df[df['severity_score'] == i]) > 0]
    labels_with_data = [severity_labels[i] for i in range(4) 
                       if len(df[df['severity_score'] == i]) > 0]
    ax7.boxplot(conf_by_severity, labels=labels_with_data)
    ax7.set_ylabel('Confidence Score (%)')
    ax7.set_title('Confidence by Severity')
    ax7.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Low threshold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    summary_text = f"""
    SUMMARY STATISTICS
    
    Valid Detections: {len(df)}
    Zero Cases: {num_zero_cases}
    
    Width (mm):
      Mean: {df['lateral_ventricle_width_mm'].mean():.2f}
      Median: {df['lateral_ventricle_width_mm'].median():.2f}
      Range: {df['lateral_ventricle_width_mm'].min():.2f} - {df['lateral_ventricle_width_mm'].max():.2f}
    
    Normal (<10mm): {len(df[df['lateral_ventricle_width_mm'] < 10])} ({len(df[df['lateral_ventricle_width_mm'] < 10])/len(df)*100:.1f}%)
    Abnormal (≥10mm): {len(df[df['lateral_ventricle_width_mm'] >= 10])} ({len(df[df['lateral_ventricle_width_mm'] >= 10])/len(df)*100:.1f}%)
    
    Mean Confidence: {df['confidence'].mean():.1f}%
    High Confidence: {df['high_confidence'].sum()} ({df['high_confidence'].sum()/len(df)*100:.1f}%)
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace')
    
    # 9. Cumulative distribution
    ax9 = plt.subplot(3, 3, 9)
    sorted_widths = np.sort(df['lateral_ventricle_width_mm'])
    cumulative = np.arange(1, len(sorted_widths) + 1) / len(sorted_widths) * 100
    ax9.plot(sorted_widths, cumulative, linewidth=2)
    ax9.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10mm threshold')
    ax9.set_xlabel('Lateral Ventricle Width (mm)')
    ax9.set_ylabel('Cumulative Percentage (%)')
    ax9.set_title('Cumulative Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Test Set Evaluation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = 'test_set_evaluation_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Visualizations saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    df, zero_cases = evaluate_test_set()
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. test_set_comprehensive_results.csv - Valid detections only")
    if zero_cases:
        print("  2. zero_ventricle_cases.csv - Cases with no ventricle detection")
    print("  3. test_set_evaluation_results.png - Visualizations")
    print("\nYou can now use these results for your final presentation!")
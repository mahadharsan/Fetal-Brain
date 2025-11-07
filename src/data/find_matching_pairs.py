"""
FIND MATCHING PAIRS SCRIPT
==========================
This script finds which ultrasound images have exact matching masks.
We'll only use these matched pairs for training to avoid misalignment.
"""

from pathlib import Path
import pandas as pd
import shutil
import os

def find_matching_pairs(ultrasound_dir, mask_dir):
    """
    Find exact matches between ultrasound images and segmentation masks.
    
    Parameters:
    -----------
    ultrasound_dir : str or Path
        Directory containing original ultrasound images
    mask_dir : str or Path
        Directory containing segmentation masks
    
    Returns:
    --------
    dict
        Dictionary with matching statistics and lists
    """
    
    # Convert to Path objects
    ultrasound_dir = Path(ultrasound_dir)
    mask_dir = Path(mask_dir)
    
    # Get all files (without extension)
    print("Step 1: Loading file lists...")
    ultrasound_files = {f.stem: f for f in ultrasound_dir.glob('*.png')}
    mask_files = {f.stem: f for f in mask_dir.glob('*.png')}
    
    print(f"   Found {len(ultrasound_files)} ultrasound images")
    print(f"   Found {len(mask_files)} mask images")
    
    # Find exact matches
    print("\nStep 2: Finding exact filename matches...")
    exact_matches = []
    ultrasound_only = []
    mask_only = []
    
    # Check each ultrasound for matching mask
    for ultrasound_name, ultrasound_path in ultrasound_files.items():
        if ultrasound_name in mask_files:
            exact_matches.append({
                'name': ultrasound_name,
                'ultrasound': ultrasound_path,
                'mask': mask_files[ultrasound_name]
            })
        else:
            ultrasound_only.append(ultrasound_name)
    
    # Find masks without ultrasounds
    for mask_name in mask_files.keys():
        if mask_name not in ultrasound_files:
            mask_only.append(mask_name)
    
    # Print statistics
    print(f"\n📊 MATCHING STATISTICS:")
    print("=" * 50)
    print(f"   ✅ Exact matches: {len(exact_matches)}")
    print(f"   ❌ Ultrasounds without masks: {len(ultrasound_only)}")
    print(f"   ❌ Masks without ultrasounds: {len(mask_only)}")
    
    # Show examples of unmatched
    if ultrasound_only:
        print(f"\n   Examples of ultrasounds without masks:")
        for name in ultrasound_only[:3]:
            print(f"      - {name}")
    
    if mask_only:
        print(f"\n   Examples of masks without ultrasounds:")
        for name in mask_only[:3]:
            print(f"      - {name}")
    
    # Try pattern matching for unmatched
    print("\nStep 3: Analyzing naming patterns...")
    patient_matches = find_patient_level_matches(ultrasound_only, mask_only, 
                                                 ultrasound_files, mask_files)
    
    return {
        'exact_matches': exact_matches,
        'ultrasound_only': ultrasound_only,
        'mask_only': mask_only,
        'patient_matches': patient_matches,
        'total_ultrasounds': len(ultrasound_files),
        'total_masks': len(mask_files)
    }

def find_patient_level_matches(ultrasound_only, mask_only, ultrasound_files, mask_files):
    """
    Try to match by patient ID when exact match fails.
    This helps understand the naming inconsistency.
    """
    patient_matches = []
    
    for u_name in ultrasound_only[:10]:  # Check first 10
        patient_id = u_name.split('_')[0]
        
        # Find masks with same patient ID
        possible_masks = [m for m in mask_only if m.startswith(patient_id)]
        
        if possible_masks:
            patient_matches.append({
                'ultrasound': u_name,
                'possible_masks': possible_masks,
                'patient_id': patient_id
            })
    
    if patient_matches:
        print("\n   🔍 Found patient-level matches (but different planes/slices):")
        for pm in patient_matches[:3]:
            print(f"      Ultrasound: {pm['ultrasound']}")
            print(f"      Could match: {pm['possible_masks'][0]}")
            print()
    
    return patient_matches

def create_matched_dataset(results, output_dir):
    """
    Create a new directory with only matched pairs.
    
    Parameters:
    -----------
    results : dict
        Results from find_matching_pairs
    output_dir : str or Path
        Where to save matched pairs
    """
    output_dir = Path(output_dir)
    
    # Create output directories
    matched_ultrasound_dir = output_dir / 'matched_ultrasounds'
    matched_mask_dir = output_dir / 'matched_masks'
    
    matched_ultrasound_dir.mkdir(parents=True, exist_ok=True)
    matched_mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStep 4: Creating matched dataset in {output_dir}")
    
    # Copy matched pairs
    for match in results['exact_matches'][:10]:  # Copy first 10 as test
        # Copy ultrasound
        shutil.copy2(match['ultrasound'], matched_ultrasound_dir / match['ultrasound'].name)
        # Copy mask
        shutil.copy2(match['mask'], matched_mask_dir / match['mask'].name)
    
    print(f"   ✅ Copied {min(10, len(results['exact_matches']))} matched pairs")
    
def save_matching_report(results, output_file='matching_report.csv'):
    """
    Save a CSV report of all matches and mismatches.
    """
    # Create DataFrame with all matches
    if results['exact_matches']:
        df_matches = pd.DataFrame([
            {
                'filename': m['name'],
                'has_ultrasound': True,
                'has_mask': True,
                'status': 'matched'
            }
            for m in results['exact_matches']
        ])
    else:
        df_matches = pd.DataFrame()
    
    # Add unmatched ultrasounds
    if results['ultrasound_only']:
        df_ultrasound = pd.DataFrame([
            {
                'filename': name,
                'has_ultrasound': True,
                'has_mask': False,
                'status': 'ultrasound_only'
            }
            for name in results['ultrasound_only']
        ])
    else:
        df_ultrasound = pd.DataFrame()
    
    # Add unmatched masks
    if results['mask_only']:
        df_mask = pd.DataFrame([
            {
                'filename': name,
                'has_ultrasound': False,
                'has_mask': True,
                'status': 'mask_only'
            }
            for name in results['mask_only']
        ])
    else:
        df_mask = pd.DataFrame()
    
    # Combine all
    df_all = pd.concat([df_matches, df_ultrasound, df_mask], ignore_index=True)
    
    # Save to CSV
    df_all.to_csv(output_file, index=False)
    print(f"\n📄 Saved detailed report to {output_file}")
    
    # Print summary statistics
    print("\n📊 FINAL SUMMARY:")
    print("=" * 50)
    print(df_all['status'].value_counts())
    
    return df_all

# MAIN EXECUTION
if __name__ == "__main__":
    """
    Run the matching analysis
    """
    print("=" * 60)
    print("FINDING MATCHING PAIRS BETWEEN ULTRASOUNDS AND MASKS")
    print("=" * 60)
    
    # Set your paths here
    ULTRASOUND_DIR = 'data/raw/trans_ventricular_original'
    MASK_DIR = 'data/raw/trans_ventricular/segmentation_mask/SegmentationClass'
    OUTPUT_DIR = 'data/processed'
    
    # Find matches
    results = find_matching_pairs(ULTRASOUND_DIR, MASK_DIR)
    
    # Save report
    df_report = save_matching_report(results, 'matching_report.csv')
    
    # Decision point
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    
    match_rate = len(results['exact_matches']) / results['total_ultrasounds'] * 100
    
    if match_rate > 80:
        print(f"✅ Good match rate ({match_rate:.1f}%)!")
        print("   You can proceed with the matched pairs only.")
    elif match_rate > 50:
        print(f"⚠️  Moderate match rate ({match_rate:.1f}%)")
        print("   You'll lose some data but should have enough for training.")
    else:
        print(f"❌ Poor match rate ({match_rate:.1f}%)")
        print("   Need to investigate the naming convention mismatch.")
    
    print(f"\nNext steps:")
    print(f"1. Review 'matching_report.csv' to see all files")
    print(f"2. Use only the {len(results['exact_matches'])} matched pairs for training")
    print(f"3. Update dataset class to load only from matched list")
    
    # Optional: Create matched dataset
    # Uncomment this to copy matched files to a new directory
    # create_matched_dataset(results, OUTPUT_DIR)
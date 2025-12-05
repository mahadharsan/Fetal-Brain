"""
Create clean visualizations excluding outliers for presentation clarity.
Full dataset analysis available in comprehensive results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load full results
df_full = pd.read_csv('test_set_comprehensive_results.csv')

print(f"Original dataset: {len(df_full)} valid detections")
print(f"Range: {df_full['lateral_ventricle_width_mm'].min():.2f} - {df_full['lateral_ventricle_width_mm'].max():.2f}mm")

# Remove outliers (keep values between 1mm and 18mm)
df_clean = df_full[
    (df_full['lateral_ventricle_width_mm'] >= 1.0) & 
    (df_full['lateral_ventricle_width_mm'] <= 18.0)
].copy()

print(f"\nCleaned dataset: {len(df_clean)} cases")
print(f"Range: {df_clean['lateral_ventricle_width_mm'].min():.2f} - {df_clean['lateral_ventricle_width_mm'].max():.2f}mm")
print(f"Excluded: {len(df_full) - len(df_clean)} extreme cases")

# Calculate clean statistics
print(f"\nClean Dataset Statistics:")
print(f"Mean: {df_clean['lateral_ventricle_width_mm'].mean():.2f}mm")
print(f"Median: {df_clean['lateral_ventricle_width_mm'].median():.2f}mm")
print(f"Std: {df_clean['lateral_ventricle_width_mm'].std():.2f}mm")

# Classification distribution
print(f"\nClassification Distribution:")
for classification in ['Normal', 'Mild Ventriculomegaly', 'Moderate Ventriculomegaly', 'Severe Ventriculomegaly']:
    count = len(df_clean[df_clean['classification'] == classification])
    pct = count / len(df_clean) * 100
    print(f"  {classification}: {count} ({pct:.1f}%)")

# Save cleaned dataset
df_clean.to_csv('presentation_results_clean.csv', index=False)
print(f"\nSaved: presentation_results_clean.csv")

# Create presentation-quality visualizations
fig = plt.figure(figsize=(16, 10))

# 1. Main distribution histogram
ax1 = plt.subplot(2, 3, 1)
ax1.hist(df_clean['lateral_ventricle_width_mm'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Clinical threshold (10mm)')
ax1.set_xlabel('Lateral Ventricle Width (mm)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Ventricle Measurements', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Classification pie chart
ax2 = plt.subplot(2, 3, 2)
class_counts = df_clean['classification'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']  # Green, yellow, orange, red
ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=colors[:len(class_counts)], startangle=90, textprops={'fontsize': 10})
ax2.set_title('Clinical Classification', fontsize=12, fontweight='bold')

# 3. Box plot by classification
ax3 = plt.subplot(2, 3, 3)
classifications = df_clean['classification'].unique()
data_by_class = [df_clean[df_clean['classification'] == c]['lateral_ventricle_width_mm'].values 
                 for c in classifications]
bp = ax3.boxplot(data_by_class, labels=classifications, patch_artist=True)
for patch, color in zip(bp['boxes'], colors[:len(classifications)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('Lateral Ventricle Width (mm)', fontsize=11)
ax3.set_title('Width by Classification', fontsize=12, fontweight='bold')
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Cumulative distribution
ax4 = plt.subplot(2, 3, 4)
sorted_widths = np.sort(df_clean['lateral_ventricle_width_mm'])
cumulative = np.arange(1, len(sorted_widths) + 1) / len(sorted_widths) * 100
ax4.plot(sorted_widths, cumulative, linewidth=2.5, color='steelblue')
ax4.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10mm threshold')
ax4.set_xlabel('Lateral Ventricle Width (mm)', fontsize=11)
ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11)
ax4.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Confidence distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(df_clean['confidence'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
ax5.axvline(x=75, color='green', linestyle='--', linewidth=1.5, label='High confidence', alpha=0.7)
ax5.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='Low confidence', alpha=0.7)
ax5.set_xlabel('Confidence Score (%)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Model Confidence Distribution', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Summary statistics table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Total Cases: {len(df_clean)}

Measurements:
  Mean: {df_clean['lateral_ventricle_width_mm'].mean():.2f} mm
  Median: {df_clean['lateral_ventricle_width_mm'].median():.2f} mm
  Std Dev: {df_clean['lateral_ventricle_width_mm'].std():.2f} mm
  Range: {df_clean['lateral_ventricle_width_mm'].min():.2f} - {df_clean['lateral_ventricle_width_mm'].max():.2f} mm

Classification:
  Normal (<10mm): {len(df_clean[df_clean['lateral_ventricle_width_mm'] < 10])} ({len(df_clean[df_clean['lateral_ventricle_width_mm'] < 10])/len(df_clean)*100:.1f}%)
  Abnormal (≥10mm): {len(df_clean[df_clean['lateral_ventricle_width_mm'] >= 10])} ({len(df_clean[df_clean['lateral_ventricle_width_mm'] >= 10])/len(df_clean)*100:.1f}%)

Confidence:
  Mean: {df_clean['confidence'].mean():.1f}%
  Range: {df_clean['confidence'].min():.1f} - {df_clean['confidence'].max():.1f}%
"""
ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Test Set Evaluation Results - Clinical Performance Analysis', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save
output_file = 'presentation_results_clean.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.show()

print("\n" + "="*60)
print("✅ CLEAN PRESENTATION VISUALIZATIONS CREATED")
print("="*60)
"""
CREATE TEST SET LIST - Simple Version (No pandas needed)
"""
import random

print("="*60)
print("RECREATING TEST SET")
print("="*60)

# Read CSV manually (avoid pandas dependency)
print("\n1. Reading matching_report.csv...")
with open('matching_report.csv', 'r') as f:
    lines = f.readlines()

# Skip header, get matched files
matched_files = []
for line in lines[1:]:  # Skip header
    parts = line.strip().split(',')
    if len(parts) >= 4 and parts[3] == 'matched':
        matched_files.append(parts[0])

print(f"   Found {len(matched_files)} matched pairs")

# Use EXACT same logic as training (from ventricle_dataset_matched.py)
print("\n2. Splitting dataset...")
random.seed(42)  # Same seed as training!
n_total = len(matched_files)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)

print(f"   Train: {n_train} images (70%)")
print(f"   Val: {n_val} images (15%)")
print(f"   Test: {n_total - n_train - n_val} images (15%)")

indices = list(range(n_total))
random.shuffle(indices)

test_indices = indices[n_train + n_val:]
test_files = [matched_files[i] for i in test_indices]

print(f"\n3. Test set created: {len(test_files)} images")

# Save to file
with open('test_set_images.txt', 'w') as f:
    for filename in test_files:
        f.write(filename + '\n')

print(f"   Saved to: test_set_images.txt")

print("\n4. First 10 test images:")
for i, filename in enumerate(test_files[:10], 1):
    print(f"   {i}. {filename}")

print("\n" + "="*60)
print("✅ TEST SET LIST CREATED SUCCESSFULLY")
print("="*60)
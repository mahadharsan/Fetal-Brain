import pandas as pd

df = pd.read_csv('data/raw/trans_ventricular/Trans-ventricular-Pixel-Size.csv')
print(f"Pixel size range: {df['Pixel in mm'].min():.3f} - {df['Pixel in mm'].max():.3f}")
print(f"Average: {df['Pixel in mm'].mean():.3f}")

# Also check for your test images
test_images = ['Patient01779_Plane3_2_of_4.png', 
               'Patient01782_Plane3_1_of_2.png',
               'Patient01785_Plane3_3_of_3.png',
               'Patient01786_Plane3_2_of_3.png',
               'Patient01789_Plane3_3_of_3.png']

print("\nPixel sizes for your test images:")
for img in test_images:
    if img in df['Label'].values:
        pixel_size = df[df['Label'] == img]['Pixel in mm'].values[0]
        print(f"{img}: {pixel_size:.3f} mm")
    else:
        print(f"{img}: NOT FOUND")
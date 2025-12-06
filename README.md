# Automated Fetal Ventriculomegaly Detection System

A deep learning-based automated system for detecting and measuring fetal ventriculomegaly in ultrasound images using dual U-Net architecture. This project addresses a critical clinical need in prenatal care by automating the measurement of lateral ventricle width in fetal brain ultrasound scans.

## Overview

Ventriculomegaly is a condition characterized by enlarged cerebral ventricles in the fetal brain, diagnosed when lateral ventricle width exceeds 10mm. This system automates the detection and measurement process, providing clinicians with accurate measurements and visual explanations to support diagnostic decisions.

### Key Features

- **Dual U-Net Architecture**: Specialized models for brain tissue and lateral ventricle segmentation
- **Clinical Accuracy**: Achieves 91.4% Dice score for brain segmentation and 71.8% for ventricle segmentation
- **Automated Measurement**: Calculates lateral ventricle width with proper pixel spacing calibration
- **Clinical Classification**: Categorizes cases based on the 10mm diagnostic threshold
- **Explainable AI**: Grad-CAM visualizations showing model attention regions
- **Confidence Scoring**: Prediction probabilities to assess measurement reliability

## Clinical Significance

The system produces clinically realistic measurements ranging from 4.33mm to 12.89mm, with proper classification into:
- **Normal**: Lateral ventricle width < 10mm (81.2% of test cases)
- **Mild Ventriculomegaly**: 10-12mm
- **Moderate Ventriculomegaly**: 12-15mm
- **Severe Ventriculomegaly**: > 15mm (18.8% detection rate across various degrees)

## Technical Architecture

### Dual U-Net Design

The system employs two specialized U-Net models to address severe class imbalance:

1. **Brain Segmentation Model**: Segments overall fetal brain tissue from background
2. **Ventricle Segmentation Model**: Focuses specifically on lateral ventricles within brain regions

This approach outperforms single multi-class models by avoiding gradient competition and decoder bottlenecks.

### Model Specifications

- **Architecture**: U-Net with encoder-decoder structure
- **Input Size**: 256×256 grayscale images
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (learning rate: 0.001)
- **Training Split**: 70% train, 15% validation, 15% test (fixed seed: 42)

### Measurement Pipeline

1. **Image Preprocessing**: Resize and normalize ultrasound images
2. **Brain Segmentation**: Identify fetal brain tissue regions
3. **Ventricle Segmentation**: Detect lateral ventricles within brain mask
4. **Measurement Extraction**: Calculate lateral ventricle width with pixel spacing calibration
5. **Clinical Classification**: Apply 10mm threshold for diagnosis
6. **Visualization**: Generate Grad-CAM heatmaps and annotated results

## Dataset

**HC18 Challenge Dataset**
- 584 ultrasound-mask pairs
- Patient-specific pixel spacing data from Trans-ventricular-Pixel-Size.csv
- Original image dimensions preserved for accurate measurement scaling

## Performance Metrics

| Metric | Brain Segmentation | Ventricle Segmentation |
|--------|-------------------|------------------------|
| Dice Score | 91.4% | 71.8% |
| IoU | 84.3% | 56.0% |
| Precision | 92.8% | 80.1% |
| Recall | 90.1% | 65.5% |

**Clinical Performance:**
- Valid measurements: 69 out of 69 quality test images (100%)
- Dataset quality issues: 20 images contained blank/corrupted data or severely degraded quality
- Confidence scoring: Failed detections showed significantly lower confidence (25% vs 56% mean)
- Grad-CAM validation: 78-80% attention-ventricle overlap on successful segmentations

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/fetal-ventriculomegaly-detection.git
cd fetal-ventriculomegaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models
```python
# Train brain segmentation model
python train_brain_model.py --epochs 50 --batch_size 16

# Train ventricle segmentation model
python train_ventricle_model.py --epochs 50 --batch_size 16
```

### Running Inference
```python
# Process test set with full pipeline
python dual_model_pipeline_LVW_gradcam_confidence.py

# Evaluate complete test set
python evaluate_full_test_set.py

# Generate clean visualizations
python create_clean_visualizations.py

# Validate Grad-CAM attention
python validate_gradcam.py
```

## Key Scripts

- `dual_model_pipeline_LVW_gradcam_confidence.py`: Complete clinical pipeline with measurement and visualization
- `evaluate_full_test_set.py`: Comprehensive evaluation metrics
- `create_clean_visualizations.py`: Generate publication-quality figures
- `validate_gradcam.py`: Validate explainability visualizations

## Critical Implementation Details

### Pixel Spacing Calibration

A crucial aspect of the system is proper pixel spacing calibration. Image resizing from original dimensions to 256×256 model input requires scaling factors that account for:
- Original image dimensions (varying across dataset)
- Patient-specific pixel spacing values
- Proper conversion from pixels to millimeters

Without this calibration, measurements are underestimated by 2-3×, potentially misclassifying borderline cases.

### Uncertainty Quantification

The system implements robust confidence scoring that successfully identifies unreliable predictions. Failed detections consistently show significantly lower confidence scores (25% vs 56% mean), allowing the system to flag cases requiring manual clinical review. This uncertainty handling is critical for safe clinical deployment.

## Limitations

- **Dataset Quality**: Some test images contained blank/corrupted data, highlighting the importance of data validation pipelines
- **Challenging Cases**: Performance varies with image quality, fetal positioning, and severe pathology
- **Dataset Size**: Limited training data may affect generalization to diverse clinical settings
- **Single View**: System analyzes single ultrasound frames rather than multi-view sequences

## Future Work

- Automated data quality filtering to identify corrupted or blank images
- Multi-view integration for improved accuracy and robustness
- Extended training on larger, more diverse datasets
- Clinical validation with radiologist annotations
- Real-time inference optimization for clinical deployment
- Longitudinal tracking across gestational ages

## Project Team

**Northeastern University - DS 5500 Capstone Project**
- Mahadharsan
- Bupesh Kumar Ramesh Kumar

## Acknowledgments

This project utilizes the HC18 Challenge dataset for fetal head ultrasound segmentation. We thank the organizers for making this valuable resource available to the research community.

## License

MIT License

Copyright (c) 2024 Mahadharsan & Bupesh Kumar Ramesh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## References

- HC18 Challenge Dataset
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

## Contact

For questions or collaboration opportunities, please open an issue or contact the project team.

---

**Note**: This system is designed for research purposes and should not be used as the sole basis for clinical decisions. All automated measurements should be reviewed by qualified medical professionals.
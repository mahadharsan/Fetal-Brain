![License](https://img.shields.io/badge/license-MIT-green.svg)

# Automated Fetal Ventriculomegaly Detection System

End-to-end solo project: data pipeline, dual U-Net modeling, measurement calibration, explainability, and evaluation.

Ventriculomegaly — enlargement of the brain's fluid-filled ventricles — is one of the most common findings on prenatal ultrasound and can signal developmental issues if missed. This system automates the measurement doctors currently do by hand, using two neural networks: one to find the brain, one to find the ventricles inside it.

Deep learning system for detecting and measuring fetal ventriculomegaly in ultrasound images using dual U-Net architecture. Automates lateral ventricle width measurement for prenatal diagnosis (diagnostic threshold: 10mm).

## Key Features
- **Dual U-Net Architecture**: Separate models for brain tissue and lateral ventricle segmentation
- **Clinical Accuracy**: 91.4% Dice score (brain), 71.8% Dice score (ventricles)
- **Automated Measurement**: Pixel spacing calibrated lateral ventricle width calculation
- **Explainable AI**: Grad-CAM visualizations with 78-80% attention-ventricle overlap
- **Confidence Scoring**: Robust uncertainty quantification (25% vs 56% mean for failed vs successful detections)

## Clinical Results
- Clinically realistic measurements: 4.33mm - 12.89mm range
- 100% valid measurements on quality test images (69/69 cases)
- Classification: 81.2% normal, 18.8% ventriculomegaly (mild/moderate/severe)

## Technical Architecture

### Dual U-Net Design
Two specialized U-Net models address severe class imbalance:
- **Brain Segmentation**: Segments fetal brain tissue from background
- **Ventricle Segmentation**: Detects lateral ventricles within brain regions

### Specifications
- Input: 256×256 grayscale images
- Loss: Binary Cross-Entropy with Logits
- Optimizer: Adam (lr: 0.001)
- Split: 70% train, 15% validation, 15% test (seed: 42)

### Pipeline
Image Preprocessing → Brain Segmentation → Ventricle Segmentation → Measurement Extraction (with pixel spacing calibration) → Clinical Classification → Grad-CAM Visualization

## Dataset
**HC18 Challenge Dataset**
- 584 ultrasound-mask pairs
- Patient-specific pixel spacing data
- Original dimensions preserved for accurate scaling

## Installation
```bash
git clone https://github.com/mahadharsan/Fetal-Brain.git
cd Fetal-Brain
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
# Complete pipeline
python dual_model_pipeline_LVW_gradcam_confidence.py

# Evaluation
python evaluate_full_test_set.py

# Visualizations
python create_clean_visualizations.py
```

## Critical Implementation: Pixel Spacing Calibration
Proper calibration accounts for:
- Original image dimensions (dataset-specific variation)
- Patient-specific pixel spacing values
- Accurate pixel-to-millimeter conversion

Without calibration, measurements are underestimated by 2-3×, risking misclassification of borderline cases.

## Limitations
- Dataset quality: 20 blank/corrupted images in test set
- Performance varies with image quality and fetal positioning
- Single-view analysis (no multi-view integration)

## Future Work
- Automated data quality filtering
- Multi-view integration
- Extended training on larger datasets
- Clinical validation with radiologist annotations

## Project Context
Built independently as part of Northeastern University's DS 5500 Capstone.

## License
MIT License - Copyright (c) 2024 Mahadharsan Ravichandran

## References
- HC18 Challenge Dataset
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Grad-CAM: Visual Explanations from Deep Networks

**Disclaimer:** Research purposes only. Not approved for clinical use. All measurements should be reviewed by qualified medical professionals.

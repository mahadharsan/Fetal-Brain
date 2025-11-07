# Fetal Ventricle Segmentation Model Evaluation Report

Generated: 2025-10-20 17:23:09

## Model Information
- Model Path: `src\training\best_model.pth`
- Device: cuda
- Test Samples: 89

## Segmentation Performance Metrics
| Metric | Mean ± Std | Median | Min-Max | 95% CI |
|--------|------------|--------|---------|--------|
| Dice | 0.7137 ± 0.3183 | 0.8473 | 0.0000-1.0000 | [0.6462, 0.7811] |
| Iou | 0.6299 ± 0.3087 | 0.7350 | 0.0000-1.0000 | [0.5645, 0.6953] |
| Sensitivity | 0.7917 ± 0.2634 | 0.8805 | 0.0000-1.0000 | [0.7359, 0.8475] |
| Specificity | 0.9987 ± 0.0020 | 0.9992 | 0.9902-1.0000 | [0.9983, 0.9992] |
| Precision | 0.8067 ± 0.2720 | 0.8824 | 0.0000-1.0000 | [0.7491, 0.8643] |

## Clinical Measurement Accuracy
- Mean Area Difference: 6907303396.75% ± 29183297655.21%

## Clinical Interpretation
- Performance Level: **Good - Suitable for clinical research and validation studies**
- Dice Score: 0.7137

## Recommendations
- Consider additional training with more diverse data
- Review cases with lowest Dice scores for failure patterns
- Low sensitivity suggests model missing some ventricle regions
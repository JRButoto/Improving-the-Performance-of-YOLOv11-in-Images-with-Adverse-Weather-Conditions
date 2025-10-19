# Improving YOLOv11 Performance in Adverse Weather Conditions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📄 Research Paper
[**Read the Full Paper (PDF)**](docs/Improving_YOLOv11_Performance_in_Adverse_Weather.pdf)

## 🎯 Project Overview

This project implements an end-to-end image-adaptive pipeline that enhances YOLOv11's object detection performance in adverse weather conditions (fog, rain, sand, and snow). The approach uses a CNN-based Parameter Predictor (CNN-PP) module with EfficientNetB0 to automatically predict optimal enhancement parameters for weather-degraded images.

### The Problem
Object detection models like YOLOv11 achieve high accuracy on standard benchmarks but degrade sharply under adverse weather conditions. Images captured in fog, rain, sand, or snow suffer from:
- Reduced contrast and visibility
- Obscured object features
- Increased noise and artifacts
- Domain shift from training data

### Our Solution
I developed a two-stage pipeline:

1. **CNN-PP Module**: A lightweight EfficientNetB0-based network that predicts six enhancement parameters:
   - Defog intensity
   - White balance (per RGB channel)
   - Gamma correction
   - Tone curve adjustments
   - Contrast scaling
   - Unsharp masking

2. **Differentiable Image Processing (DIP)**: Applies the predicted parameters to enhance degraded images before detection

### Key Findings

#### CNN-PP Performance
- Achieved extremely low prediction error (MAE < 0.003) on synthetic data
- Successfully learned to map degraded images to optimal enhancement parameters

#### Detection Results (mAP@0.5)

| Weather Condition | Baseline YOLOv11 | Enhanced YOLOv11 | Change |
|-------------------|------------------|------------------|---------|
| Fog               | 0.75            | 0.40             | -0.35   |
| Rain              | 0.55            | 0.61             | +0.06 ✓ |
| Sand              | 0.67            | 0.38             | -0.29   |
| Snow              | 0.63            | 0.57             | -0.06   |

The pipeline showed improvement in rainy conditions but mixed results in others, highlighting the challenge of transferring synthetic training to real-world weather artifacts.

## 🔬 Technical Approach

### Dataset Usage
- **COCO Dataset**: 20,151 images used to train CNN-PP on synthetic degradations
- **DAWN Dataset**: 5,553 real adverse weather images for evaluation

### Pipeline Architecture
1. Input degraded image (200×200) → CNN-PP
2. CNN-PP outputs 6 filter parameter groups
3. DIP module applies filters in sequence
4. Enhanced image → YOLOv11 for detection

### Implementation Details
- **CNN-PP Backbone**: Frozen EfficientNetB0 features + parameter-specific heads
- **Training**: 100 epochs with MSE loss on synthetic paired data
- **Filter Ranges**: Carefully tuned based on weather degradation characteristics

## 🚧 Limitations & Insights

The project revealed important challenges in weather-adaptive object detection:

1. **Domain Gap**: Synthetic training doesn't fully capture real weather complexity
2. **Precision-Recall Trade-off**: Enhancement improved precision but often reduced recall
3. **Weather Specificity**: One-size-fits-all filters struggle with diverse conditions

These findings suggest future work should focus on:
- Domain adaptation techniques
- Weather-specific enhancement models
- Learnable filter architectures

## 📧 Contact

John Butoto - [johnbutoto@gmail.com](mailto:johnbutoto@gmail.com)

LinkedIn: [John R. Butoto](https://linkedin.com/in/jrbutoto/)
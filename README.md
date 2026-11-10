# NeuroScan AI: Clinical Brain Tumor MRI Diagnostic Pipeline

Clinical Decision Support System (CDSS) powered by ResNet-50, Automated Otsu Skull-Stripping, and Domain Adaptation across heterogeneous multi-scanner MRI data.

## Verified Consolidated Benchmark (N = 1,914 Patient Scans)
- **Consolidated Accuracy:** 94.67%
- **Meningioma Recall:** 97.37% (Precision: 91.81%)
- **Healthy Brain (No Tumor) Recall:** 99.38%
- **Pituitary Tumor Recall:** 99.78% (Precision: 96.59%)
- **Glioma Recall:** 82.29% (Precision: 97.53%)

## Key Modules
1. `pipelines/clinical_pipeline.py`: Otsu skull-stripper and Class-Weighted Focal Loss.
2. `models/Clinical_Brain_ResNet50_Calibrated.pth`: Domain-adapted weights with temperature calibration (T = 1.4089).
3. `results/`: Multi-center confusion matrices and validation metrics.

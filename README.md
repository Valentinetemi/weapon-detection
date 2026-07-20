# Weapon Detection POC

A proof-of-concept object detection system for identifying handguns, rifles, and knives in public settings, built with YOLOv8. Developed as a technical assessment for Startup Campus / Schuele's Computer Vision team.

## Overview

This project demonstrates whether modern object detection models can reliably identify weapons using publicly available data, as a foundation for a larger security/situational-awareness detection system.

## Repository Structure

```
weapon-detection/
├── README.md                     # this file
├── Weapon_model.ipynb            # full Colab notebook (data merge, training, evaluation)
├── data/
│   └── merged_dataset_backup/    # annotated, balanced dataset (YOLOv8 format)
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml             # class names + dataset config
├── report/
│   ├── technical_report.md       # full technical report
│   └── images/                   # confusion matrix, sample detections, training curves
└── weight/
    └── best_weapon_model.pt      # final trained YOLOv8 model weights
```

## Quick Summary

- **Classes:** Guns, Rifle, knife
- **Dataset:** ~149 images merged from 4 public sources (Roboflow Universe), balanced across classes with a manual priority pass favoring more realistic/diverse imagery
- **Model:** YOLOv8n (Ultralytics), trained on Google Colab (T4 GPU)
- **Final performance:** mAP50 = 0.532, mAP50-95 = 0.276 (see [technical report](report/technical_report.md) for full breakdown and iteration history)

## Model Weights

Trained weights are included directly in this repository: [`weight/best_weapon_model.pt`](weight/best_weapon_model.pt)

## Dataset

The annotated, merged, balanced dataset (YOLOv8 format) is included directly in this repository: [`data/merged_dataset_backup/`](data/merged_dataset_backup/)

## Full Report

See [report/technical_report.md](report/technical_report.md) for:
- Dataset preparation approach
- Training configuration
- Detection performance (including mAP, confusion matrix)
- Sample detection results
- Challenges encountered
- Recommendations for scaling to production

## Author

Temiloluwa Valentine Olajuwon

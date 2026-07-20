# Technical Report: Weapon Detection Proof of Concept

**Author:** Temiloluwa Valentine Olajuwon
**Date:** July 2026

## 1. Objective

Build a proof-of-concept object detection model capable of identifying handguns, rifles, and knives in public settings, to evaluate whether modern object detection models (YOLOv8) can reliably support a future production weapon detection system.

## 2. Dataset Preparation

### 2.1 Sources

Four public datasets were sourced from Roboflow Universe and merged into a single unified dataset:

| Source Dataset | Original Classes | Contribution |
|---|---|---|
| weapon-detection--4 | Guns, Rifle | Base handgun/rifle imagery |
| Knife-Detection-1 | knife | Base knife imagery |
| real-world-gun-data-2 | gun, human | Realistic/CCTV-style gun imagery (human class dropped) |
| Knife-detection-1 (realistic) | knife | Realistic/dynamic knife imagery (self-defense/combat scenes) |

### 2.2 Class Schema

All source datasets used different class orderings and, in some cases, different class scopes. These were remapped to a single unified 3-class schema:

```
0: Guns
1: Rifle
2: knife
```

Where a source dataset included a "human" class (real-world-gun-data-2), it was intentionally excluded from the merge, since person detection was out of scope for this assignment.

### 2.3 Balancing Strategy

Rather than using all available images from each source (some sources had 700–2,500+ images), a deliberate subsampling approach was used:

1. Images were bucketed by which class(es) they contained (an image can contain more than one weapon).
2. A target count per class was selected and images were randomly sampled up to that target, ensuring no single class dominated the dataset.
3. Where a source dataset was judged to be more "realistic" (varied context, dynamic poses, CCTV-style angles) versus "clean" (studio/product-style shots), that source was prioritized during sampling to favor diversity over raw volume.

Initial balancing targeted ~50 images per class. After an initial training run revealed Rifle as the weakest-performing class, the Rifle target was increased and the dataset was rebalanced and regenerated before a second training run (see Section 4).

### 2.4 Final Dataset Composition (Iteration 2)

- **Total images:** ~161
- **Train / Valid / Test split:** 70% / 20% / 10%
  - Train: 125 images
  - Valid: 36 images
  - Test: ~16-19 images

A 70/20/10 split was chosen over the source datasets' default splits (which were often heavily train-skewed, e.g. 96/0/4) to ensure a meaningful validation set for monitoring training, given the small overall dataset size.

## 3. Training Configuration

- **Model:** YOLOv8n (Ultralytics), pretrained weights as starting point
- **Image size:** 640x640
- **Batch size:** 16
- **Epochs:** 100 (with early stopping, patience=20)
- **Optimizer:** AdamW (auto-selected), lr0 ≈ 0.0014
- **Augmentations:** Default Ultralytics augmentation pipeline (mosaic, flip, HSV jitter, blur, CLAHE, etc.)
- **Hardware:** Google Colab, Tesla T4 GPU (free tier)
- **Training time:** ~3-5 minutes per run

## 4. Detection Performance

### 4.1 Iteration 1 (Baseline)

Initial training used a balanced dataset with ~50 images per class (149 total images, 104 train).

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Guns | 0.503 | 0.290 | 0.385 | 0.165 |
| Rifle | 0.474 | 0.265 | 0.228 | 0.082 |
| knife | 0.605 | 0.615 | 0.613 | 0.311 |
| **Overall** | **0.527** | **0.390** | **0.409** | **0.186** |

Rifle was clearly the weakest-performing class, both in detection rate (recall) and localization accuracy (mAP).

### 4.2 Iteration 2 (After Rebalancing)

Diagnosis: Rifle had the least diverse source data (limited to one dataset, predominantly military/tactical imagery). The per-class sampling target for Rifle was increased, pulling in more volume from the existing source, and the full pipeline (merge → balance → split → train) was re-run.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Guns | 0.722 | 0.667 | 0.632 | 0.224 |
| Rifle | 0.509 | 0.346 | 0.336 | 0.204 |
| knife | 1.000 | 0.478 | 0.628 | 0.401 |
| **Overall** | **0.744** | **0.497** | **0.532** | **0.276** |

**Result:** Overall mAP50 improved from 0.409 → 0.532 (+30%), and mAP50-95 improved from 0.186 → 0.276 (+48%). Rifle mAP50 improved from 0.228 → 0.336, though it remains the weakest class relative to Guns and knife.

*(See `report/images/results.png` for training/validation curves.)*

### 4.3 Confusion Matrix Analysis

*(See `report/images/confusion_matrix.png`)*

The confusion matrix from the final model reveals:

- **Guns:** 4 correct detections, 1 missed (classified as background). Clean performance overall.
- **Rifle:** 12 correct detections, but 10 missed entirely (false negatives) and 21 false positives (background misclassified as Rifle). This is by far the largest source of error in the model.
- **knife:** 4 correct detections, with a smaller number of false positives from background.

Rifle is clearly the model's weakest point — both under-detecting real rifles and over-predicting rifles where none exist. This is consistent with Rifle having the narrowest diversity of source imagery (predominantly outdoor/military/tactical contexts only).

## 5. Sample Detection Results

*(See `report/images/` for full-size versions)*

1. **Success case — Handgun (high confidence):** Model correctly identified a handgun with 0.69 confidence, tightly bounding the weapon.
2. **Miss case — Low-light CCTV footage:** Model failed to detect a person in a grainy, low-light street CCTV frame. No training data closely resembled this scenario.
3. **Failure case — Rifle:** Model missed an obvious rifle held by a person in a well-lit indoor scene, instead producing a low-confidence (0.29) false positive on an unrelated wall picture. This illustrates the Rifle class's current unreliability, consistent with the confusion matrix findings.

## 6. Challenges Encountered

- **Small dataset size:** With ~125 training images across 3 classes, the model has limited ability to generalize, particularly for less-represented visual contexts.
- **Class imbalance in source data:** Public datasets varied significantly in size and class coverage; some (e.g. the original Rifle source) offered narrow contextual diversity (mostly military/tactical scenes), directly limiting real-world generalization for that class.
- **Inconsistent class taxonomies across sources:** Source datasets used different class names, orders, and scopes (e.g. one dataset combined "gun" and "human" in a single schema), requiring careful remapping to avoid silent labeling errors.
- **Small validation set:** At this dataset scale, validation metrics can be somewhat noisy epoch-to-epoch; results should be interpreted as directional rather than highly precise.
- **Realistic vs. clean imagery tradeoff:** Many public datasets favor clean, staged, or studio-style images, which do not fully represent the visual conditions (low light, occlusion, distance, motion blur) relevant to a real security/CCTV use case.

## 7. Recommendations for Scaling to Production

1. **Scale dataset size substantially.** Production-grade weapon detection typically requires thousands of images per class, versus the ~50-80 used here. This is the single highest-leverage improvement available.
2. **Prioritize Rifle-class diversity specifically.** Given it is the current weak point, sourcing rifle imagery across a wider range of contexts (civilian, urban, varied lighting/distance) would likely yield the largest single performance gain, as confirmed by the iteration 1 → 2 improvement.
3. **Expand class granularity.** Current scope (Guns, Rifle, knife) could be extended in future iterations to include additional weapon subtypes (e.g. distinguishing handgun types, adding blunt weapons or explosives) as more labeled data becomes available.
4. **Incorporate real CCTV/security footage.** Since the target deployment context is public security systems, training data should include more low-light, distant, and low-resolution imagery representative of actual security camera feeds, rather than relying primarily on clean or staged photography.
5. **Increase validation/test set size** as more data becomes available, to produce more statistically reliable performance estimates.
6. **Consider a larger model variant** (YOLOv8s/m) once dataset size justifies it — the current YOLOv8n was chosen for speed given the small-data regime, but a larger model may better exploit a bigger dataset.
7. **Establish an iterative data-collection loop**, similar to the process used here: deploy, monitor per-class failure patterns via a confusion matrix, and target new data collection at the weakest-performing classes.

## 8. Conclusion

This proof of concept demonstrates that a YOLOv8-based model can be trained to detect weapons (Guns, Rifle, knife) from a modest public dataset, achieving an overall mAP50 of 0.532 after one round of targeted iteration. While not yet at production-grade accuracy, the results validate the overall approach and pipeline, and the analysis above identifies clear, evidence-based priorities (primarily dataset scale and Rifle-class diversity) for improving performance toward a deployable system.
# Technical Report: Weapon Detection Proof of Concept

**Author:** Temiloluwa Valentine Olajuwon
**Date:** July 2026

## 1. Objective

The goal of this project was to find out whether a modern object detection model, trained on a small amount of publicly available data, could reliably detect weapons, specifically handguns, rifles, and knives in realistic, public-facing settings. This isn't meant to be a production-ready system. It's a proof of concept: something to show that the underlying approach works, to surface where it currently struggles, and to give a clear, evidence-based sense of what it would take to get from here to something deployable.

## 2. Dataset Preparation

### 2.1 Where the data came from

I sourced four public datasets from Roboflow Universe rather than relying on a single source, mainly because no single dataset covered all three weapon classes with the kind of visual variety I wanted. The four sources were:

- **weapon-detection--4** — a base dataset already split into Guns and Rifle classes
- **Knife-Detection-1** — a clean, single-class knife dataset
- **real-world-gun-data-2** — a more realistic, CCTV/tactical-style gun dataset, which also included a "human" class I didn't need
- **Knife-detection-1** (note the lowercase "d" — a separate dataset from the one above) — a knife dataset with more dynamic, in-context imagery, including people actively holding or wielding knives

Each of these datasets used its own class ordering, and in one case, a class I didn't want at all (the "human" label from the real-world gun dataset). Before any of this data could be combined, I had to carefully check each dataset's `data.yaml` to see exactly how its classes were indexed, then remap everything into one consistent schema:

```
0: Guns
1: Rifle
2: knife
```

This mattered more than it might seem, Roboflow doesn't always order classes the way you'd expect just from looking at the website, so skipping this step could easily have resulted in silently mislabeled data (e.g. rifles being trained as knives). I verified each mapping manually before merging anything.

### 2.2 Why I didn't just use everything

Several of these source datasets were much larger than what I needed, one had over 2,500 images, another over 1,000. Using all of it wasn't the right move. Beyond the practical issue of training time, a bigger risk was class imbalance: if I'd pulled everything, I likely would have ended up with hundreds of gun images and comparatively few knife images, which would have biased the model toward detecting guns well and knives poorly, regardless of how good the knife data itself was.

So instead, I built a small pipeline that:

1. Scanned every image and recorded which class(es) it actually contained (some images have more than one weapon in frame)
2. Grouped images into per-class "buckets"
3. Randomly sampled a target number of images from each bucket, so no single class dominated the final dataset
4. Where I had two datasets covering the same class (as with knife), I gave priority to the more visually realistic one the one with varied poses, angles, and contexts over the cleaner, more "product photo"-style one, so that when the sampler had to choose, it favored realism over just volume

The result was a dataset that stayed small and balanced by design, rather than large and skewed by accident.

### 2.3 Final dataset and split

After two rounds of sampling (explained in Section 4), the final merged dataset contained roughly 161 images, split as follows:

- **Train:** 125 images
- **Valid:** 36 images
- **Test:** 19 images

That's a 70/20/10 split. I chose this deliberately over the default splits that came with the source datasets. One of them, for example, put 96% of its images in "train" and almost nothing in "valid," which would have made it very hard to tell during training whether the model was actually learning or just memorizing. Given how small my overall dataset was to begin with, having a meaningful validation set felt more important than maximizing training volume.

## 3. Training Configuration

I trained using YOLOv8n (the "nano" variant) via Ultralytics, starting from its pretrained weights rather than training from scratch. Given the small size of my dataset, a smaller model made more sense cause it trains fast, it's less prone to overfitting on limited data, and speed mattered also.

Key settings:
- Image size: 640×640
- Batch size: 16
- Epochs: 100, with early stopping (patience of 20 epochs — training stops if validation performance doesn't improve for 20 epochs in a row)
- Optimizer: AdamW, automatically tuned by Ultralytics
- Augmentations: Ultralytics' default pipeline (mosaic, horizontal flip, HSV color jitter, blur, etc.)
- Hardware: Google Colab's free-tier Tesla T4 GPU

Each full training run took only a few minutes, which made it realistic to iterate more than once — which turned out to matter a lot, as the next section explains.

## 4. Detection Performance

### 4.1 First attempt

My first training run used a balanced dataset with about 50 images per class (149 images total, 104 for training). The results were a reasonable starting point, but far from even:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Guns | 0.503 | 0.290 | 0.385 | 0.165 |
| Rifle | 0.474 | 0.265 | 0.228 | 0.082 |
| knife | 0.605 | 0.615 | 0.613 | 0.311 |
| **Overall** | **0.527** | **0.390** | **0.409** | **0.186** |

Rifle stood out immediately as the weak point, both in how often the model found rifles at all (recall of 0.265) and in how well it localized them (mAP50 of 0.228, less than half of knife's score). Looking back at the dataset, this made sense: Rifle was the one class that came from only a single source, and that source leaned heavily toward one visual context outdoor, military, tactical scenes. The model simply hadn't seen enough variety to generalize.

### 4.2 Diagnosing and fixing it

Rather than guessing at hyperparameter tweaks, I went back to the data. I increased the sampling target specifically for Rifle, pulling in more images from the existing source, rebalanced the full dataset, redid the train/valid/test split, and retrained from scratch. The dataset grew slightly (161 images total, 125 for training), but the real change was in the *composition*, not just the count.

The second run showed a clear improvement across every class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Guns | 0.722 | 0.667 | 0.632 | 0.224 |
| Rifle | 0.509 | 0.346 | 0.336 | 0.204 |
| knife | 1.000 | 0.478 | 0.628 | 0.401 |
| **Overall** | **0.744** | **0.497** | **0.532** | **0.276** |

Overall mAP50 went from 0.409 to 0.532 about a 30% relative improvement and mAP50-95 (a stricter metric that weighs bounding box accuracy more heavily) improved by nearly 50%. Rifle's mAP50 improved from 0.228 to 0.336. It's still the weakest of the three classes, but the gap closed meaningfully, and it confirmed that the issue really was about data diversity, not something fundamentally wrong with the model or pipeline.

Interestingly, Guns improved by the largest margin of any class (mAP50 up by 0.247), even though I hadn't specifically added more Guns data. I think this is simply because the overall training set got bigger and more varied, and that lifted every class somewhat, a reminder that in small-data regimes, general dataset growth helps even the classes you weren't directly targeting.

### 4.3 What the confusion matrix shows

Looking at the confusion matrix provides a clearer picture of where the model succeeds and where it struggles, beyond the overall performance metrics.

Guns: There were 6 actual gun images in the test set. The model correctly detected 4 of them but missed 2, predicting them as background. It also incorrectly classified 1 background image as a gun.

Rifle: There were 33 actual rifle images. The model correctly detected 12, but it missed 21, predicting them as background instead. It also incorrectly classified 10 background images as rifles. This indicates that the model struggles both to recognize rifles consistently and to distinguish them from background scenes.

Knife: There were 10 actual knife images. The model correctly detected 4 but missed 6, again predicting them as background. Unlike the other classes, it did not incorrectly classify any background images as knives.

Overall, the Rifle class remains the model's weakest area. Not only did it have the highest number of missed detections (21), but it also produced the largest number of false positives (10). This suggests that the model has not yet learned a robust representation of rifles across different environments and viewpoints. The failure example shown in the sample detections, where the model completely missed an obvious rifle in a well-lit image actually supports this observation. Collecting more diverse rifle images, particularly from civilian and low-light settings, would likely lead to the greatest improvement in future versions of the model.

## 5. Sample Detection Results

Running the final model on the held-out test set produced a mix of results, which I think is actually the most honest way to represent where this POC stands:

**A clean success.** On one test image, the model detected a handgun with 0.69 confidence, with the bounding box tightly and correctly placed around the weapon. This is the kind of result that shows the underlying approach works when the conditions resemble the training data.

**A near-miss in a hard scenario.** On a grainy, low-light CCTV-style street image, the model detected nothing at all. There's a person visible in the frame, but between the low resolution, the dim lighting, and the distance from the camera, the model simply had no comparable training examples to draw on. This wasn't unexpected, it's a good illustration of the type of imagery a security-camera deployment would actually need to handle, and a gap in the current training data.

**An outright, and honestly kind of funny, failure.** In one indoor test image, a person is holding a rifle plainly in frame unmistakable to a human eye. The model missed it completely. Instead, it flagged a small framed picture on the wall as "Guns" with a low 0.29 confidence score. This is a clear example of the Rifle-class weakness surfaced by the confusion matrix not just a rare edge case, but a real failure on an easy, well-lit example.

I'm including all three of these deliberately, rather than only the success case, because I think the failure cases are actually more informative about what needs to happen next.

## 6. Challenges Encountered

A few things made this project harder than it might look from the outside:

- **The dataset is just small.** 125 training images across three classes is nowhere near what a production model would need. Everything else in this report, the imbalance issues, the Rifle weakness — traces back to this constraint.
- **Public datasets don't agree with each other.** Different class names, different orderings, different scopes (one dataset lumped in a "human" class I didn't want). Merging them safely required careful, manual verification at every step rather than assuming things would line up.
- **Rifle's source data was narrow.** It wasn't that the Rifle images were bad, they just all came from a similar visual context (military/tactical), which limited how well the model could generalize to other settings.
- **The validation set is small enough that metrics have some noise.** With only 36 validation images, individual epoch-to-epoch metrics bounced around a fair amount during training. The overall trend across the full run is trustworthy; any single epoch's number should be read with a bit of caution.
- **Most public weapon datasets are "clean."** A lot of available images are staged, well-lit, or product-style photos. Real security footage looks nothing like that — low light, motion blur, distance, partial occlusion and there's very little public data that reflects those conditions well.

## 7. Recommendations for Scaling to Production

If this were to move toward an actual production system, here's where I'd focus, roughly in order of impact:

1. **Get more data — a lot more.** This is, by far, the single biggest lever. Production-grade detection models are typically trained on thousands of images per class, not tens. Everything else on this list matters less than this.
2. **Prioritize Rifle-class diversity specifically.** The data backs this up directly — Rifle is the weakest class, and it's the class with the least varied source data. Sourcing rifle imagery from more civilian, urban, and varied-lighting contexts would likely produce the single largest improvement available, based on what we saw between the first and second training runs.
3. **Bring in real security-camera-style footage.** Since the end goal is a public security system, the training data should include more low-light, distant, and lower-resolution imagery that actually resembles what a CCTV feed looks like — not just clear daytime photos.
4. **Expand the class taxonomy over time**, once there's enough data to support it — splitting knife types, or adding categories like blunt weapons or explosives, could come later without needing to redesign the pipeline.
5. **Grow the validation and test sets** as more data becomes available, so performance numbers become more statistically reliable and less sensitive to any one batch of images.
6. **Consider a larger model (YOLOv8s or YOLOv8m)** once the dataset justifies it. I deliberately used the smallest variant here because the dataset was small and speed mattered for this assignment, but a bigger model would likely make better use of a bigger dataset.
7. **Keep the iteration loop going.** The approach that worked here — train, look closely at the confusion matrix, find the weakest class, go get more of exactly that kind of data, retrain is a repeatable process, not a one-time fix. I'd expect that same loop, applied a few more times, to steadily close the remaining gaps.

## 8. Conclusion

This proof of concept shows that a YOLOv8-based model, even trained on a fairly small and carefully curated public dataset, can learn to detect weapons across three categories with reasonable though not yet production-ready accuracy. The final model reached an overall mAP50 of 0.532, up from 0.409 in an earlier iteration, after diagnosing and directly addressing a specific weakness in the Rifle class.

More important than the raw number, I think, is that the process behind it is sound and repeatable: source data carefully, balance it deliberately, train, evaluate honestly (including looking at what the model gets wrong, not just what it gets right), and use that evaluation to decide what to fix next. That loop is what would need to run several more times, with meaningfully more data, to get this from a working proof of concept to something ready for real-world deployment.

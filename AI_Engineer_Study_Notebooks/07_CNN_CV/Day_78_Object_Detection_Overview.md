# Day 78: Object Detection Overview

## 1. Beyond Classification
Up until now, we’ve asked: "**What is in this image?**" (Classification).
Now we ask: "**Where is it?**" (Localization) and "**What are they?**" (Object Detection).

- **Classification**: "Cat" (1 label per image)
- **Object Detection**: "Cat at [x,y,w,h]", "Dog at [x,y,w,h]" (Multiple boxes per image)

---

## 2. Key Concepts

### 2.1 Bounding Box (BBox)
Defined by 4 numbers: `[x_center, y_center, width, height]` or `[x_min, y_min, x_max, y_max]`.

### 2.2 IoU (Intersection over Union)
How do we grade a box? We compare it to the Ground Truth (GT).

$$ IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$

- **IoU = 1.0**: Perfect match.
- **IoU > 0.5**: Generally considered a "Correct" detection.
- **IoU < 0.5**: Miss.

### 2.3 NMS (Non-Maximum Suppression)
Detectors often output 50 boxes for the same cat. NMS cleans this up:
1. Sort boxes by confidence score.
2. Pick the best box.
3. Discard all other boxes that overlap heavily (high IoU) with the best box.
4. Repeat.

---

## 3. The Two Families of Detectors

### 3.1 Two-Stage Detectors (R-CNN Family)
"Accuracy First, Speed Second"

**Steps:**
1. **Region Proposal**: Find "blobs" that look like objects (Potential candidates).
2. **Classification**: Run a CNN on *each* candidate to identify it.

**Evolution:**
- **R-CNN**: Slow. running CNN 2000 times per image.
- **Fast R-CNN**: Smarter. Run CNN once, chop features.
- **Faster R-CNN**: Replaced external region proposal with a "Region Proposal Network" (RPN). **Standard for high accuracy.**

### 3.2 One-Stage Detectors (YOLO Family)
"Speed First"

**YOLO (You Only Look Once)** treats detection as a **single regression problem**.
1. Divide image into a grid (e.g., $7 \times 7$).
2. Each cell predicts BBox coordinates and Class probabilities directly.

**Pros**: Extremely fast (Real-time, 45+ FPS).
**Cons**: Historically struggled with small objects (though YOLOv8+ is amazing).
**Versions**: YOLOv1 -> YOLOv3 -> YOLOv5 -> YOLOv8 (Ultralytics).

---

## 4. Comparison

| Feature | Faster R-CNN (Two-Stage) | YOLO (One-Stage) |
|:--------|:-------------------------|:-----------------|
| **Speed** | Slow (~5 FPS) | Fast (30-100 FPS) |
| **Accuracy**| High (especially small objects) | Good (SOTA is very close now) |
| **Use Case**| Medical Imaging, Surveillance | Self-driving cars, Robotics |

---

## 5. Summary
- **Classification vs Detection**: "What" vs "What + Where".
- **IoU**: The metric for box accuracy.
- **NMS**: Removing duplicate boxes.
- **R-CNN**: Two-stage, accurate, slow.
- **YOLO**: One-stage, fast, real-time.

**Next Up:** **Likely a Project** or **Semantic Segmentation**.

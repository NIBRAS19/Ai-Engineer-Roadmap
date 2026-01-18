# Day 69: Introduction to Convolutional Neural Networks (CNNs)

## 1. Why Vanilla Neural Networks Fail on Images
If you take a $224 \times 224$ image, flatten it ($50,176$ pixels), and feed it to a Dense Layer with 100 neurons:
- Weights = $50,176 \times 100 \approx 5 \text{ Million}$ parameters.
- **Problem 1**: Huge memory usage.
- **Problem 2**: Loss of Spatial Structure. The network doesn't know that "pixel A" is next to "pixel B".

## 2. The Solution: Convolution
Instead of looking at the whole image at once, we look at **patches**.
We slide a small "Filter" (kernel) over the image to detect features.
- First layers find simple lines/edges.
- Middle layers find eyes/ears.
- Final layers find faces/cats.

---

## 3. The Core Components
1.  **Convolutional Layer**: Feature extractor.
2.  **Pooling Layer**: Reduces size (Downsampling).
3.  **Fully Connected Layer**: Classifier at the end.

---

## 4. Summary
- **CNNs**: Preserves spatial relationships.
- **Parameter Sharing**: The same "Edge Detector" filter works on the top-left and bottom-right of the image.

**Next Up:** **Convolutional Layers**—The math behind the sliding window.

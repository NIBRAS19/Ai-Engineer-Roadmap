# Day 41: K-Means Clustering

## 1. Introduction
Unsupervised Learning. No labels (y).
We want to group similar items together.
- Customer Segmentation (High Spenders, Window Shoppers).
- Image Compression (Group similar colors).

---

## 2. The Algorithm
1.  Choose K (number of clusters).
2.  Randomly place K "Centroids".
3.  Assign every point to closest Centroid.
4.  Move Centroid to the average of its points.
5.  Repeat until they stop moving.

---

## 3. Implementation

```python
from sklearn.cluster import KMeans

# We don't have y here!
model = KMeans(n_clusters=3)
model.fit(X)

labels = model.predict(X)       # [0, 1, 0, 2...] (Cluster IDs)
centroids = model.cluster_centers_
```

---

## 4. Choosing K (Elbow Method)
How many clusters do we need?
Plot **Inertia** (Sum of squared distances) vs K.
Look for the "Elbow" where improvements slow down.

---

## 5. Practical Exercises

### Exercise 1: Color Compression
1.  Load an image.
2.  Treat every pixel (R,G,B) as a data point.
3.  Run K-Means with K=16.
4.  Replace every pixel with its centroid.
Result: The image uses only 16 colors.

---

## 6. Summary
- **Clustering**: Grouping without labels.
- **Centroids**: The center of a cluster.
- **K**: Must be chosen manually.

**Next Up:** **PCA**—Compressing data.

# Day 87: Word Embeddings

## 1. The Idea
Instead of a sparse vector of size 100,000 (One-Hot), what if we represent every word as a **Dense Vector** of size 300?
Crucially, **Similar words should have similar vectors.**
- King - Man + Woman $\approx$ Queen.

---

## 2. Word2Vec (2013)
Google's breakthrough.
- **Hypothesis**: "You shall know a word by the company it keeps." (Distributional Semantics).
- **Skip-Gram**: Predict context words given a center word. (Input: "Fox", Target: "Quick", "Brown", "Jumps").
- **CBOW**: Predict center word given context.

## 3. GloVe (Global Vectors)
Stanford's alternative which uses matrix factorization of co-occurrence counts.

---

## 4. PyTorch Implementation (`nn.Embedding`)
In deep learning, we learn embeddings from scratch alongside our model.

```python
import torch
import torch.nn as nn

# Vocab Size = 1000, Embedding Dim = 50
embedding_layer = nn.Embedding(1000, 50)

# Input: Indices of words (e.g., [1, 45, 23])
input_indices = torch.LongTensor([1, 45])
vector = embedding_layer(input_indices)

print(vector.shape) # (2, 50)
```

---

## 5. Summary
- **Embeddings**: Dense representations where math works ($distance(cat, dog) < distance(cat, car)$).
- **Pre-trained**: You can download Google's Word2Vec or GloVe and load them into your `nn.Embedding` layer.

**Next Up:** **RNNs**—Handling sequences of text.

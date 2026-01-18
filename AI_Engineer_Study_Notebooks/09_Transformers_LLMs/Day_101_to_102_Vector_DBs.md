# Days 101-102: Embeddings and Vector Databases

## 1. The Search Problem
SQL databases search for exact keyword matches.
"Laptop" won't match "Notebook Computer".
**Semantic Search** solves this by searching for **Meaning** (Vectors).

---

## 2. Embeddings
We convert text to a vector (e.g., list of 768 numbers) using an OpenAI API or a local BERT model.
- $Similarity(A, B) = Cosine(Vector_A, Vector_B)$

---

## 3. Vector Databases
Calculating similarity between 1 query and 1 Million documents takes too long.
**Vector DBs** (Pinecone, Chroma, FAISS, Weaviate) index vectors for ultra-fast "Approximate Nearest Neighbor" search.

---

## 4. Implementation (FAISS)
Facebook AI Similarity Search.

```python
import faiss
import numpy as np

# 1. Database of 1000 vectors (dim=64)
db_vectors = np.random.random((1000, 64)).astype('float32')

# 2. Build Index (L2 Distance)
index = faiss.IndexFlatL2(64)
index.add(db_vectors)

# 3. Query
query_vector = np.random.random((1, 64)).astype('float32')
distances, indices = index.search(query_vector, k=5) # Find top 5 closest
```

---

## 5. Summary
- **Embed**: Convert text to logic.
- **Index**: Store logic in Vector DB.
- **Search**: Retrieve logic.

**Next Up:** **Prompt Engineering**—Speaking to the AI.

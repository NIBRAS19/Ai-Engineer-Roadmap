# Day 86: Bag of Words and TF-IDF

## 1. Bag of Words (BoW)
How do we convert "I love AI" into a vector?
We create a **Vocabulary** of all unique words in our dataset.
Each sentence is a vector of counts.

Sentence 1: "I love cats"
Sentence 2: "I love dogs"
Vocab: [I, love, cats, dogs]

Vector 1: [1, 1, 1, 0]
Vector 2: [1, 1, 0, 1]

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love AI", "AI is great"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out()) # ['ai', 'great', 'is', 'love']
print(X.toarray()) 
# [[1, 0, 0, 1],
#  [1, 1, 1, 0]]
```

---

## 2. TF-IDF
**Term Frequency - Inverse Document Frequency**.
- **Issue with BoW**: Common words like "the", "good" appear everywhere and have high counts, but imply little specific meaning.
- **Solution**: Penalize words that appear in *many* documents.
    - High TF-IDF = Word is frequent in THIS document, but rare elsewhere (Important Keyword).
    - Low TF-IDF = Word is distinct or common everywhere.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)
```

---

## 3. Summary
- **Sparse Vectors**: BoW and TF-IDF create huge vectors (size of vocab, e.g., 100,000) mostly filled with zeros.
- **No Semantics**: "Good" and "Great" are equidistant. The model doesn't know they are synonyms.

**Next Up:** **Word Embeddings**—Solving the synonym problem.

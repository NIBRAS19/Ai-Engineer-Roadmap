# Day 96: BERT (Bidirectional Encoder Representations from Transformers)

## 1. Introduction
BERT (2018) is an **Encoder-Only** model.
It is designed to **understand** text, not generate it.
Perfect for: Classification, Search (Embeddings), Q&A.

---

## 2. Pre-training Objectives
How did Google train BERT on the whole internet without labels?
It used **Self-Supervised Learning**.

### 2.1 Masked Language Modeling (MLM)
"The cat sat on the [MASK]." -> Model attempts to predict "mat".
It learns context from *both* left and right directions (Bidirectional).

### 2.2 Next Sentence Prediction (NSP)
Given Sentence A and Sentence B, does B logically follow A?

---

## 3. Usage (Embeddings)

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

# The "Last Hidden State" is the vector representation of the text
embeddings = outputs.last_hidden_state
print(embeddings.shape) 
# (1, 3, 768) -> Batch=1, Tokens=3 (Hello, world, CLS), Dim=768
```

---

## 4. Summary
- **CLS Token**: Special token added to start. Its vector represents the *entire sentence*.
- **Encoder**: Great at looking at the whole picture.

**Next Up:** **GPT**—The Generator.

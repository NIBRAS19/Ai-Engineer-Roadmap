# Day 94: Hugging Face Ecosystem

## 1. Introduction
Implementing a Transformer from scratch in PyTorch is hard (hundreds of lines of math).
**Hugging Face (HF)** provides the `transformers` library.
It's the "GitHub of AI" + "Scikit-Learn of Deep Learning".

- **Transformers Library**: Code.
- **Model Hub**: 500,000+ pre-trained models.
- **Datasets Library**: Standardized data loading.

---

## 2. The Pipeline API
The easiest way to use AI.

```python
# pip install transformers
from transformers import pipeline

# 1. Sentiment Analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I love this course!")) 
# [{'label': 'POSITIVE', 'score': 0.99}]

# 2. Text Generation
generator = pipeline("text-generation", model="gpt2")
print(generator("The future of AI is", max_length=20))
```

---

## 3. Behind the Pipeline
The pipeline does 3 things:
1.  **Tokenizer**: Text -> Numbers.
2.  **Model**: Numbers -> Logits.
3.  **Post-Processing**: Logits -> Labels ("POSITIVE").

---

## 4. Summary
- **Pipeline**: Zero-code inference.
- **Hub**: Always check huggingface.co/models before training your own.

**Next Up:** **Tokenizers**—How text becomes numbers.

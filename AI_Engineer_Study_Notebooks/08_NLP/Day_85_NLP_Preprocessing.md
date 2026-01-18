# Day 85: Introduction to NLP and Preprocessing

## 1. What is NLP?
Natural Language Processing (NLP) is the field of teaching computers to understand, interpret, and generate human language.
**Challenges**:
- Ambiguity ("I saw the man with the telescope").
- Sarcasm.
- Slang/Idioms.

---

## 2. Text Preprocessing Pipeline
Computers cannot understand raw string "Hello world". We must clean it.

### 2.1 Lowercasing & Punctuation Removal
Standardizing text.

### 2.2 Tokenization
Splitting text into words (tokens).
"I love AI" -> ["I", "love", "AI"]

### 2.3 Stop Words Removal
Removing common words that carry little meaning (the, is, at, which).

### 2.4 Stemming vs Lemmatization
Reducing words to their root.
- **Stemming**: Chopping off ends (Running -> Run, Better -> Bet). Crude.
- **Lemmatization**: Using a dictionary to find the root (Better -> Good, Running -> Run). Slower but accurate.

---

## 3. Implementation (NLTK & SpaCy)

```python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "The quick brown foxes are jumping over the lazy dogs."
doc = nlp(text)

# Tokenization + Lemmatization + Stopwords
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

print(tokens) 
# ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog']
# Note: 'foxes' became 'fox', 'jumping' became 'jump', 'dogs' became 'dog'.
```

---

## 4. Summary
- **Garbage In, Garbage Out**: If you don't clean text, your model learns noise.
- **SpaCy**: The industry standard for robust preprocessing.

**Next Up:** **Bag of Words**—Turning text into numbers.

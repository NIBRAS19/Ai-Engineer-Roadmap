# Day 95: Tokenizers (BPE & WordPiece)

## 1. The OOV Problem
In old word-based models, if "Unstoppable" isn't in the vocab, we get `<UNK>`.
We can't have a vocab of infinite size.

## 2. Subword Tokenization
Break words into meaningful chunks.
- "Unstoppable" -> "Un" + "##stop" + "##able"
- "Playing" -> "Play" + "##ing"

If we have seen "Play", "Stop", "Able", "Un", "Ing" before, we can represent NEW words composed of them.
**Virtually eliminates Unknown tokens.**

---

## 3. Algorithms
1.  **BPE (Byte Pair Encoding)**: Used by GPT. Merges most frequent pairs of characters.
2.  **WordPiece**: Used by BERT. Similar to BPE but maximizes likelihood.

---

## 4. Implementation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Transformers are amazing!"

# 1. Tokenize
tokens = tokenizer.tokenize(text)
print(tokens) 
# ['transform', '##ers', 'are', 'amazing', '!']

# 2. Convert to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids) 
# [10938, 2546, 2024, 6429, 999]

# 3. Decode
decoded = tokenizer.decode(ids)
print(decoded) # "transformers are amazing !"
```

---

## 5. Summary
- **Subwords**: The standard for modern NLP.
- **AutoTokenizer**: Automatically loads the correct tokenizer for a given model name.

**Next Up:** **BERT**—The Encoder King.

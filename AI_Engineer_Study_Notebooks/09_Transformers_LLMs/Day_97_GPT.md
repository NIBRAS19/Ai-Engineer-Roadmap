# Day 97: GPT (Generative Pre-trained Transformer)

## 1. Introduction
GPT is a **Decoder-Only** model.
It is designed to **predict the next word**.
"The cat sat on the" -> "mat".

---

## 2. Causal Language Modeling (CLM)
Unlike BERT (which sees future words), GPT can only see **Past** words.
It uses **Masked Self-Attention** to prevent cheating (looking ahead).

---

## 3. The Scaling Laws
OpenAI found that performance scales predictably with:
1.  Model Size (Parameters).
2.  Dataset Size (Tokens).
3.  Compute (FLOPS).

**GPT-1**: 110M params.
**GPT-2**: 1.5B params.
**GPT-3**: 175B params (Emergent abilities like coding appeared here).
**GPT-4**: Trillions (estimated).

---

## 4. Usage (Generation)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_ids = tokenizer.encode("The theory of relativity is", return_tensors='pt')

# Generate
output = model.generate(input_ids, max_length=20, num_return_sequences=1)

print(tokenizer.decode(output[0]))
```

---

## 5. Summary
- **Autoregressive**: Generates one token, adds it to input, generates next.
- **Decoder**: Cannot see the right side (Future).

**Next Up:** **Fine-Tuning**—Customizing these giants.

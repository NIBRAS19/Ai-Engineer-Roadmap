# Day 113: LLM Optimization & Inference Speed

## 1. The Inference Bottleneck
Training LLMs is hard, but **serving** them is harder.
- **Latency**: Users hate waiting 5 seconds for a word.
- **Memory**: A 70B parameter model in float16 needs ~140GB VRAM. (An A100 has 80GB).
- **Cost**: Running GPUs 24/7 is expensive.

How do we make LLMs fast and cheap?

---

## 2. KV Cache (Key-Value Cache)
The **#1 Reason** why GPT is fast.

### The Problem
In autoregressive generation (predicting token $t+1$), the model needs to attend to all previous tokens $0...t$.
Re-calculating the **Key** and **Value** vectors for "Hello world" every single time we generate a new word is redundant. "Hello world" doesn't change!

### The Solution
We **Cache** the K and V matrices of past tokens in GPU memory.
- **Without Cache**: $O(n^2)$ compute per token.
- **With Cache**: $O(n)$ compute per token (we only compute K/V for the *new* token).

**Trade-off**: Increases VRAM usage (memory) to save Compute (speed).

```python
# Conceptual PyTorch code
# Past_key_values stores the cache
output = model(input_ids, past_key_values=cache)
new_token = output.logits.argmax()

# Update cache with new token's KV
new_cache = output.past_key_values
```

---

## 3. Quantization (Making Models Smaller)
Standard weights are **FP16** (16-bit Floating Point).
- 1 Parameter = 2 Bytes.
- 7B Model = 14GB VRAM.

### 3.1 INT8 (8-bit)
Reduces size by 2x (7B -> 7GB). negligible accuracy loss.
Uses `LLM.int8()` technique to handle outliers.

### 3.2 4-bit (QLoRA / GPTQ / AWQ)
Reduces size by 4x (7B -> 3.5GB).
Now a 7B model fits on a consumer GPU (RTX 3060)!
- **GPTQ**: Post-training quantization (fast inference).
- **AWQ**: Activation-aware (better accuracy).

```python
# Loading a model in 4-bit using BitsAndBytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", 
    quantization_config=nf4_config
)
```

---

## 4. Flash Attention (Making Attention Fast)
Standard Attention is Memory Bound ($O(N^2)$).
**Flash Attention (Dao et al. 2022)** optimizing GPU hardware (SRAM vs HBM).
- It breaks attention calculation into small tiles that fit in the GPU's fast cache (SRAM).
- **Result**: 2-4x speedup, massive reduction in memory usage.
- Allows context windows of 100k+ tokens.

*Note: You usually just enable this with `attn_implementation="flash_attention_2"` in HuggingFace.*

---

## 5. Context Window Management
LLMs have a limit (e.g., 4k, 8k, 128k tokens).

### Strategies for Long Documents:
1.  **Stuffing**: Just cram it in until it breaks (bad).
2.  **Map-Reduce**: Summarize chunks separately, then summarize the summaries.
3.  **Refine**: Ask the LLM to refine an answer sequentially reading chunks.
4.  **RAG**: Retrieve only relevant chunks (The Gold Standard).

---

## 6. Summary
- **KV Cache**: Essential for fast text generation.
- **Quantization (4-bit)**: Essential for running locally.
- **Flash Attention**: Essential for long context.

**Next Up:** **AI Agents**—Using these optimized LLMs to use tools.

# Day 93: The Transformer Architecture

## 1. "Attention Is All You Need" (2017)
The paper that changed everything.
It proposed getting rid of Recurrence (RNNs) and Convolution completely.
Instead, use **Self-Attention** and **Feed-Forward Networks** stacked on top of each other.

---

## 2. The Core Concept: Self-Attention

### 🎯 Real-World Analogy: The Cocktail Party Effect
> Imagine you're at a crowded party trying to understand a story your friend is telling.
> 
> - **Self-Attention** is your brain scanning every person's previous statements to understand what is being said *right now*.
> - When your friend says **"I can't believe she did that,"** your brain deeply "attends" to whoever **"she"** refers to (mentioned 5 minutes ago), while "masking out" the irrelevant noise from other conversations.
> 
> RNNs listen word-by-word and forget. Transformers listen to the **entire history at once**.

### The Math: Q, K, V
Every word gets 3 vectors:
- **Query (Q)**: What I am looking for? (e.g., "I am a pronoun looking for my noun")
- **Key (K)**: What I contain? (e.g., "I am the name 'Alice'")
- **Value (V)**: The actual information content.

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$
- $QK^T$: Dot product measures similarity. High score = "Pay attention to this word!"
- $softmax$: Turn scores into probabilities (sum to 1).
- $\times V$: Extract information from the relevant words.

---

## 3. Other Critical Components

### 3.1 Multi-Head Attention
Why one attention? Why not 8?
- Head 1 focuses on **grammar** (subject-verb agreement).
- Head 2 focuses on **entities** (names, places).
- Head 3 focuses on **sentiment** (happy/sad words).
The model learns these different "views" automatically.

### 3.2 Position-Wise Feed-Forward Networks (FFN)
After mixing information via Attention, each token processes the info **individually**.
$$ FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2 $$
*Analogy: After hearing the party gossip (Attention), you take a moment to think about it yourself (FFN).*

### 3.3 Residual Connections & Layer Normalization
The secret sauce that allows Deep Learning to work.
$$ Output = LayerNorm(x + Sublayer(x)) $$
- **Add**: The "Highway" for gradients (Solves vanishing gradient).
- **Norm**: Stabilizes training (Keeps numbers distinct).

---

## 4. Positional Encoding
Since there is no recurrence, the model doesn't know "Dog ate Man" vs "Man ate Dog".
We inject a vector representing the **Position** (Index 1, 2, 3...) into the embedding.
- Sine/Cosine functions are used so the model can extrapolate to longer lengths.

---

## 5. Architecture Structure
- **Encoder Stack**: Processes input unmasked (sees whole sentence). Used for **Understanding** (BERT).
- **Decoder Stack**: Processes output masked (cannot see future). Used for **Generation** (GPT).

| Type | Model Example | Use Case |
| :--- | :--- | :--- |
| **Encoder-Only** | BERT, RoBERTa | Classification, Search, NER |
| **Decoder-Only** | GPT-4, Llama 3 | Text Generation, Chatbots |
| **Encoder-Decoder** | T5, BART | Translation, Summarization |

---

## 6. Summary
- **Attention**: Weighted average of input based on relevance.
- **Parallelism**: Process whole sentence at once (unlike RNNs).
- **Residuals + Norm**: Key to training deep networks.

**Next Up:** **Hugging Face**—The library that democratized Transformers.


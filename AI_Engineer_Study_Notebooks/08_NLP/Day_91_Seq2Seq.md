# Day 91: Seq2Seq (Encoder-Decoder)

## 1. Introduction
Text Classification maps Many-to-One (Sentence -> Label).
Translation (English to French) maps **Many-to-Many** with **different lengths**.
- Input: "I love cats" (3 words)
- Output: "J'aime les chats" (4 words)

The standard RNN can't handle mismatched lengths easily. We need **Seq2Seq**.

---

## 2. The Architecture: Encoder-Decoder
Two RNNs (usually LSTMs or GRUs) joined together.

### 2.1 The Encoder
- Reads the input sentence one word at a time.
- Output: A single **Context Vector** ( The final hidden state $h_t$).
- This vector must contain the **entire meaning** of the sentence.

### 2.2 The Decoder
- Takes the Context Vector as its initial hidden state.
- Generates the output sentence one word at a time.
- **Input at first step**: `<SOS>` (Start of Sentence) token.
- **Stop condition**: When it generates `<EOS>` (End of Sentence).

### 🎯 Visualizing the Flow
```
       ENCODER                         DECODER
   "I"  "Love"  "Cats"           <SOS> "J'aime" "les"  "chats"
    |      |       |               |       |      |       |
  [RNN]→ [RNN]→ [RNN] ──(Context)→[RNN]→ [RNN]→ [RNN]→ [RNN]
                                       ↓       ↓      ↓
                                   "J'aime"  "les"  "chats"
```

---

## 3. Training: Teacher Forcing

How do we train the Decoder?
In inference (real usage), the decoder uses its *own predicted word* as input for the next step.
But if it predicts a wrong word early ("I love" -> "Je deteste"), the whole sentence effectively crashes.

**Teacher Forcing** solves this during training.
- We don't feed the decoder's own prediction.
- We feed the **Ground Truth** (the actual correct word) from the target sentence.
- This acts like a teacher correcting a student immediately after a mistake, rather than letting them fail for an hour.

| Mode | Input at Step $t$ | Pros | Cons |
|:-----|:------------------|:-----|:-----|
| **Inference** | Prediction from $t-1$ | Real-world usage | Errors compound (exposure bias) |
| **Teacher Forcing** | Ground Truth at $t-1$ | Faster convergence | Model might become lazy |

**Code Concept:**
```python
# Teacher Forcing Ratio = 0.5 means use ground truth 50% of the time
use_teacher_forcing = random.random() < teacher_forcing_ratio

if use_teacher_forcing:
    next_input = target_tensor[t] # Correct word
else:
    next_input = decoder_prediction # Model's own guess
```

---

## 4. Decoding Strategies (Inference)

When the model predicts "J'aime", it outputs probabilities for all 10,000 words in the vocabulary. How do we choose?

### 4.1 Greedy Search
Pick the **highest probability** word at every step.
- Fast.
- **Problem**: Missing the best overall sentence because of one sub-optimal choice early on.
- *Analogy*: Walking through a maze and always taking the widest path immediately, even if it leads to a dead end.

### 4.2 Beam Search
Keep track of the **top k** (beam width) most likely partial sentences at every step.
- **Beam Width = 3**: Keep the top 3 best sentence starts.
- At the next step, expand all 3 and keep the top 3 best *new* combinations.
- **Result**: Finds better sentences than Greedy, but costs more computation.

---

## 5. The Bottleneck Problem
The Encoder must squash the entire meaning of a long article into a **single vector**.
- If the sentence is 5 words, it works fine.
- If the sentence is 100 words, the vector "forgets" the beginning inputs.
- Performance degrades rapidly with sequence length.

**The Solution:** We need a way to let the Decoder "look back" at specific words in the source sentence when generating the translation.

---

## 6. Summary
- **Encoder-Decoder**: The standard architecture for Translation, Summarization, and Chatbots.
- **Teacher Forcing**: Using ground truth to stabilize training.
- **Beam Search**: A smarter way to generate sentences than just picking the best word every time.
- **Bottleneck**: Compressing everything into one vector is hard.

**Next Up:** **Attention**—The solution to the bottleneck and the birth of Transformers.


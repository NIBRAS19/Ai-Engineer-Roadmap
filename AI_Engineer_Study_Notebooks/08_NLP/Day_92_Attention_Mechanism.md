# Day 92: The Attention Mechanism

## 1. The Intrusion
"Why squash the whole sentence into one vector?"
**Attention** allows the Decoder to "look back" at the ENTIRE source sentence at every step.

---

## 2. How it Works (Bahdanau Attention)
When generating the word "chats" (in French):
1.  The model calculates an **Attention Score** for every word in the input ("I", "love", "cats").
2.  It finds that "cats" is most relevant.
3.  It uses a weighted average of the input states, focusing heavily on "cats".
4.  It generates "chats".

---

## 3. Impact
- **Bottleneck Removed**: No matter how long the sentence, we can look at any part of it.
- **Interpretability**: We can visualize the **Attention Weights** (Heatmap) to see what the model is looking at.

---

## 4. Looking Ahead
In 2017, Google asked: "If we have Attention, do we even need the RNNs?"
Answer: No. **Attention Is All You Need.**
This led to the **Transformer**, which we study next week.

**CONGRATULATIONS!** You have completed **Weeks 13-14: NLP**.
You moved from counting words (Bag of Words) to translating languages (Seq2Seq).
**Next Week:** **Transformers & LLMs**—BERT, GPT, and the modern AI revolution.

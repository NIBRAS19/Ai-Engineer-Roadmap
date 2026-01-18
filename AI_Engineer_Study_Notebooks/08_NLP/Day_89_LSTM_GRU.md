# Day 89: Long Short-Term Memory (LSTM)

## 1. Introduction
Designed specifically to solve the Vanishing Gradient problem.
LSTMs add a **Cell State** ($c_t$) that acts as a "super-highway" for information to flow unchanged.
They use **Gates** to control flow:
1.  **Forget Gate**: What to throw away from old memory?
2.  **Input Gate**: What to store in new memory?
3.  **Output Gate**: What to output to the next state?

---

## 2. GRU (Gated Recurrent Unit)
A simplified LSTM. Merges Cell/Hidden state.
Often performs just as well but is faster.

---

## 3. Implementation (`nn.LSTM`)

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(3, 5, 10)

# Output contains all hidden states
# hidden contains (h_n, c_n) -> Final Hidden State and Final Cell State
output, (h_n, c_n) = lstm(x)
```

---

## 4. Bidirectional LSTM
Reads text Forwards AND Backwards.
Ideally captures context from both past and future.
`bidirectional=True`. Output size doubles.

---

## 5. Summary
- **LSTM**: The gold standard for pre-Transformer NLP.
- **Gates**: Learn what to remember and what to forget.

**Next Up:** **Text Classification Project**—Sentiment Analysis using LSTM.

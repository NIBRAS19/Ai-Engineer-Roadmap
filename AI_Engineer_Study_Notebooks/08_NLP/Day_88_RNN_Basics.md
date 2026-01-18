# Day 88: Recurrent Neural Networks (RNNs)

## 1. The Problem with Feed-Forward Networks
"The food was good, not bad." vs "The food was bad, not good."
BoW sees the same words.
Standard NNs expect fixed input size. Sentences have variable length.

---

## 2. RNN Architecture
Process tokens one by one, maintaining a **Hidden State** (Memory).
$$ h_t = \tanh(W_h h_{t-1} + W_x x_t) $$
Output depends on current word $x_t$ AND previous memory $h_{t-1}$.

---

## 3. PyTorch Implementation (`nn.RNN`)

```python
# Input Size: 10 (Embedding Dim)
# Hidden Size: 20 (Memory vector size)
# Batch First: True (Batch, Seq, Feature)
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# Fake batch: 3 sentences, 5 words each, embedding dim 10
x = torch.randn(3, 5, 10) 

output, hn = rnn(x)

print(output.shape) # (3, 5, 20) -> The hidden state at every step
print(hn.shape)     # (1, 3, 20) -> The FINAL hidden state (Summary of sentence)
```

---

## 4. The Vanishing Gradient Problem
When backpropagating through time (BPTT), gradients are multiplied repeatedly.
If weights are < 1, gradient vanishes to 0. The model forgets early words.
"**The man** who ate the apple .... and drove the car ... **is** happy."
The RNN forgets "The man" by the time it reaches "is".

---

## 5. Summary
- **RNN**: Sequential processing with memory.
- **Flaw**: Cannot handle long sequences due to vanishing gradients.

**Next Up:** **LSTM**—The solution.

# Day 90: Project - Text Classification with LSTM

## 1. Goal
Classify movie reviews (IMDB) as Positive or Negative.

---

## 2. The Model Architecture

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. Dense Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text: [batch, sent_len]
        embedded = self.embedding(text)
        
        # lstm output: [batch, sent_len, hidden]
        # hn: [1, batch, hidden] (The last state)
        output, (hn, cn) = self.lstm(embedded)
        
        # We use the final hidden state to classify the sentence
        # hn[-1] extracts the last layer's state
        return self.fc(hn[-1])
```

---

## 3. Handling Variable Lengths (Padding)
Sentences have different lengths.
We must pad them to the max length in the batch using `<PAD>` token (Index 0).
Use `nn.utils.rnn.pack_padded_sequence` for efficiency (tells LSTM to ignore padding).

---

## 4. Summary
- **Embedding -> LSTM -> Fully Connected**.
- This is the classic NLP architecture pattern.

**Next Up:** **Seq2Seq**—Translation.

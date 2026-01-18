# Days 99-100: Project - Fine-Tuning BERT for Sentiment Analysis

## 1. Goal
We will fine-tune a pre-trained **BERT (DistilBERT)** model to classify movie reviews (IMDb) as Positive or Negative.
This is the "Hello World" of NLP Engineering.

**What we are building**:
- Input: "This movie was a total disaster."
- Output: `Negative (99%)`

---

## 2. The Setup

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load Data (Example)
data = [
    ("I love this movie", 1),
    ("This is terrible", 0),
    ("Best film ever", 1),
    ("Waste of time", 0)
] * 100  # Duplicate for demo
df = pd.DataFrame(data, columns=['text', 'label'])

# 2. Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class IMDbDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)
train_dataset = IMDbDataset(train_texts, train_labels)
val_dataset = IMDbDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

---

## 3. The Model
We use `DistilBertForSequenceClassification`, which adds a simple Linear Layer on top of the `[CLS]` token output.

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

---

## 4. The Training Loop
This is standard PyTorch, but with Hugging Face's computed loss.

```python
epochs = 3

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        # Move to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass (HF models return loss automatically if labels are provided)
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
```

---

## 5. Evaluation & Inference

```python
model.eval()
text = "The cinematography was stunning, but the plot was boring."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()

print("Prediction:", "Positive" if predicted_class_id == 1 else "Negative")
```

---

## 6. Project Checklist

- [ ] **Data Prep**: Clean text, split train/test. 
- [ ] **Tokenization**: Handle padding and truncation (crucial for BERT).
- [ ] **DataLoader**: Efficient batching.
- [ ] **Fine-Tuning**: Run the loop on GPU (Colab is free!).
- [ ] **Evaluation**: Check Accuracy/F1 Score.

**Challenge**: Deploy this to Hugging Face Spaces using Gradio!

---

## 7. Summary
- **Transfer Learning**: We took a model trained on Wikipedia (BERT) and taught it Movie Reviews in 5 minutes.
- **Hugging Face classes**: `ForSequenceClassification` handles the head for you.
- **Tokenizer**: Matches text to the model's vocabulary.

**Next Up:** **Vector Databases**—Giving LLMs long-term memory.


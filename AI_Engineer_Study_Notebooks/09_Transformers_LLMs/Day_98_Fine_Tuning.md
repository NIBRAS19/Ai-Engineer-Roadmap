# Day 98: Fine-Tuning with Trainer API

## 1. Introduction
Pre-trained models (Foundation Models) are generalists.
**Fine-Tuning** adapts them to a specific task (e.g., Legal text classification) using a labeled dataset.
We adjust the weights slightly.

---

## 2. The Hugging Face `Trainer`
Writing a training loop (like we did in PyTorch) for Transformers is complex (Gradient Accumulation, Schedulers, Mixed Precision).
The `Trainer` class handles all of this.

---

## 3. Implementation Steps

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# 1. Load Model with a Classification Head (2 classes)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 2. Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch"
)

# 3. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 4. Train
trainer.train()
```

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)
Fine-tuning 175B parameters is impossible on consumer hardware.
Techniques like **LoRA (Low-Rank Adaptation)** freeze the main model and only train tiny adapter layers (0.1% of params).
This allows fine-tuning Llama-2 on a single GPU.

---

## 5. Summary
- **Trainer API**: The standard way to train in HF.
- **PEFT/LoRA**: The modern way to tune LLMs.

**Next Up:** **Application**—Building a Sentiment Classifier.

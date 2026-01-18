# Days 103-105: Prompt Engineering

## 1. Introduction
Modern LLMs (GPT-4) are so smart that we often don't need to train them. We just need to ask nicely.
**Prompt Engineering** is the art of formulating the request to get the best output.

---

## 2. Techniques

### 2.1 Zero-Shot
Just asking.
"Translate to French: Hello."

### 2.2 Few-Shot (In-Context Learning)
Giving examples.
"Translate to French:
Dog -> Chien
Cat -> Chat
Hello -> "
This dramatically improves performance.

### 2.3 Chain of Thought (CoT)
Asking the model to "think step by step".
Math problems: "If I have 5 apples..."
Without CoT: Fails.
With CoT: "First, I have 5 apples settings..." -> Succeeds.

### 2.4 System Prompts
"You are an expert Data Scientist. Answer only in Python code."

---

## 3. The "Lost in the Middle" Phenomenon
LLMs pay most attention to the **Beginning** and **End** of a prompt.
Crucial information should not be buried in the middle of a long document.

---

## 4. Summary
- **Iterate**: Prompting is trial and error.
- **Be Specific**: Ambiguous prompt = Ambiguous answer.
- **Structure**: Use Markdown/headers in your prompt to separate instructions from data.

**Next Up:** **RAG**—Giving the LLM your private data.

# Days 106-107: RAG (Retrieval Augmented Generation)

## 1. The Problem: "Hallucinations & Cutoffs"
LLMs have two fatal flaws:
1.  **Knowledge Cutoff**: They don't know events after training (e.g., today's stock price).
2.  **Hallucination**: They confidently make up facts.

**RAG** connects the LLM to your **Private Methods** (PDFs, SQL, Notion).
It allows the LLM to "open a book" before answering.

---

## 2. The Architecture Steps

### Step 1: Ingestion & Chunking (The Garbage-In-Garbage-Out Phase)
You can't just feed a 500-page PDF. You must break it into **Chunks**.

**Chunking Strategies:**
- **Fixed Size**: Every 500 characters. Simple, but might cut sentences in half.
- **Recursive**: Split by paragraphs, then sentences. (Best Default).
- **Semantic**: Split where the *topic* changes (Advanced).

*Tip: Always include "Overlap" (e.g., 50 tokens) so context isn't lost at the boundaries.*

### Step 2: Embedding
Convert text chunks into Vectors (Lists of numbers).
- **Model Choice**:
    - `text-embedding-3-small` (OpenAI): Cheap, very good.
    - `bge-m3` or `e5-large` (Open Source): Free, run locally, arguably better performance.

### Step 3: Retrieval (Vector Search)
User Question: "What is the policy?" -> Vector DB finds top 3 similar chunks.

### Step 4: Generation (The Prompt)
We construct a prompt:
```text
System: You are a helpful assistant. Answer ONLY using the provided context.
Context: [Chunk 1] ... [Chunk 2] ... [Chunk 3]
User Question: What is the policy?
```

---

## 3. Why RAG > Fine-Tuning?
Beginners always ask: *"Should I fine-tune Llama 3 on my company documents?"*
**Answer: PROBABLY NO.**

- **RAG**: Like giving the model a textbook. It can cite sources. Accuracy is high. Easy to update (just replace PDF).
- **Fine-Tuning**: Like teaching the model a new speaking style. It learns *form*, not *facts*. Hard to update.

---

## 4. Key Metrics for RAG
1.  **Retrieval Accuracy**: Did we find the right document? (Hit Rate @ K).
2.  **Generation Faithfulness**: Did the LLM stick to the document or hallucinate?

---

## 5. Summary
- **Retrieval**: Finding the needle in the haystack.
- **Augmentation**: Stuffing the needle into the Prompt.
- **Generation**: LLM summarizes the answer.

**Next Up:** **The RAG Project**—Building a Chatbot.


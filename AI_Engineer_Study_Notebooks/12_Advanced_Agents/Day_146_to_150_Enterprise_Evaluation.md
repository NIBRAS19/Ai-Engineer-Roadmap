# Day 146-150: Enterprise Evaluation & Ops

## 1. The "Vibe Check" Problem
Most people test agents by "vibes" (chatting with it).
**Enterprise Engineering** requires metrics. You cannot ship what you cannot measure.

---

## 2. Frameworks for Evaluation
### 2.1 RAGAS (RAG Assessment)
- **Faithfulness**: Did the answer come from the context?
- **Answer Relevance**: Did it answer the user's question?
- **Context Recall**: Did we retrieve the right documents?

### 2.2 DeepEval
Unit tests for LLMs.
```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

test_case = LLMTestCase(input="...", actual_output="...")
metric = HallucinationMetric(threshold=0.5)
metric.measure(test_case)
```

---

## 3. Tracing & Observability
### LangSmith (LangChain)
- Visualizing the entire chain of thought.
- Seeing exactly where the agent failed (e.g., "Tool Error" vs "Bad Reasoning").
- **Dataset Creation**: One-click "Add to Dataset" from production logs.

---

## 4. Evaluation Driven Development (EDD)
1.  Create a "Golden Dataset" of 50 Q&A pairs.
2.  Run your agent against the dataset.
3.  Modify prompt/RAG parameters.
4.  Re-run. Did the score go up?
5.  **Commit**.

---

## 5. Summary
- **Observability**: Seeing what happened (LangSmith).
- **Evaluation**: Scoring how well it happened (RAGAS/DeepEval).
- **CI/CD for AI**: Running these tests automatically on GitHub Actions.

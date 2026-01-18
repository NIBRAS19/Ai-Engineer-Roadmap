# Day 151-155: Cognitive Architectures

## 1. Beyond "Prompting"
A script is static. A "Cognitive Architecture" models how a mind works.
It draws inspiration from Cognitive Science (Short-term vs Long-term memory, Executive function).

---

## 2. Key Architectures
### 2.1 Generative Agents (The "Sims" Paper)
*Park et al. (Stanford/Google)*
- **Memory Stream**: A list of *everything* the agent has ever perceived.
- **Retrieval**: Fetching relevant memories based on Recency, Importance, and Relevance.
- **Reflection**: Periodically summarizing memories to form "High-Level Thoughts".
- **Planning**: Creating a schedule based on goals.

### 2.2 Tree of Thoughts (ToT)
Instead of a straight line, the agent explores multiple possibilities.
- **Thought Generator**: Generate 3 possible next steps.
- **State Evaluator**: Rate each step (0-10).
- **Search Algorithm**: BFS (Breadth-First Search) or DFS to find the best path.

---

## 3. Tool-Use Architectures
### 3.1 ReAct (Review)
Reason -> Act -> Observe.

### 3.2 Plan-and-Solve
1.  **Planner**: Detailed step-by-step plan.
2.  **Solver**: Execute the plan. (Better for math/coding).

---

## 4. Implementation
Building a "Memory/Reflection" module in Python.
```python
class MemoryStream:
    def __init__(self):
        self.memories = []
    
    def add_memory(self, text, importance):
        self.memories.append({"text": text, "importance": importance, "time": now()})
    
    def retrieve(self, query):
        # Calculate scores based on Recency, Relevance, Importance
        return best_memories
```

---

## 5. Summary
- **Cognitive Archecture**: Giving the agent "Psychology".
- **Generative Agents**: The gold standard for social simulation.
- **Tree of Thoughts**: The gold standard for complex problem solving.

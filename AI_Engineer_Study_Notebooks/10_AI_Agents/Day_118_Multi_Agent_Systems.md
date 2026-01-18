# Day 118: Orchestration & Multi-Agent Systems

## 1. The Manager Problem
When you have 5 agents (Coder, Tester, Designer...), who talks to whom?
If everyone talks to everyone, it's chaos ($N^2$ connections).
We need **Orchestration**.

---

## 2. Orchestration Patterns

### 2.1 Sequential Handoffs (The Relay Race)
Agent A $\to$ Agent B $\to$ Agent C.
- **Example**: Researcher finds data $\to$ Writer summarizes it $\to$ Editor fixes style.
- **Pros**: Simple, predictable.
- **Cons**: Rigid. If A fails, the chain breaks.

### 2.2 Hierarchical (The Boss)
A **Routing Agent** (or Manager) decides who works next.
- **Manager**: "This is a coding task. @Coder, take this."
- **Coder**: "Done."
- **Manager**: "Now @Tester, check it."
- **Pros**: Flexible. The Manager handles edge cases.

---

## 3. Modern Frameworks

### 3.1 LangGraph (State Machines)
Replaces "Chain of Thought" with a **State Graph**.
- **Nodes**: Agents or Functions.
- **Edges**: Logic (If-Then). "If code fails test, go back to Coder. Else, go to Deploy."
- **Cyclic**: It allows **loops** explicitly! (Build $\to$ Test $\to$ Fix $\to$ Build...).

*LangGraph is the industry standard for production agents because it is **controllable**.*

### 3.2 AutoGen (Conversation)
"Just put them in a room and let them talk."
- You define generic agents.
- They chat until they agree on a solution (Termination Condition).
- **Pros**: Easy to start, emergent behavior.
- **Cons**: Hard to control, can get stuck in loops.

---

## 4. Example: LangGraph Concept

```python
# Concept Code
graph.add_node("planner", planner_agent)
graph.add_node("coder", coder_agent)
graph.add_node("tester", tester_agent)

# Define Logic
graph.add_edge("planner", "coder")
graph.add_conditional_edges(
    "coder",
    should_test_or_fix, # Function deciding next step
    {
        "test": "tester",
        "fix": "coder"   # Loop back!
    }
)
```

---

## 5. Summary
- **Orchestration**: Defining the flow of communication.
- **LangGraph**: Control flow using Graphs (Mental model: State Machine).
- **AutoGen**: Control flow using Conversation (Mental model: Meeting Room).

**Next Up:** **Safety & Guardrails**—Keeping agents in check.


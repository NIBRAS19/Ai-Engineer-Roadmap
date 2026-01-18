# Day 117: Agentic Planning Patterns

## 1. The Limit of ReAct
ReAct (Think $\to$ Act) is great for short tasks.
But for complex goals ("Write a video game"), it effectively "sleepwalks".
It only looks one step ahead. It forgets the big picture.

---

## 2. Pattern 1: Plan-and-Execute
Separate the **Planner** from the **Executor**.

### Phase 1: Planning
**Input**: "Write a snake game in Python."
**Planner Agent**:
1. Create game window.
2. Create snake class.
3. Create food class.
4. Handle collision logic.
5. Add scoring.

### Phase 2: Execution
**Executor Agent** takes Step 1: "Create game window".
- Performs ReAct loop.
- Returns result.
**Executor Agent** takes Step 2...

**Why?** The Planner maintains the global goal. The Executor focuses on the details.

---

## 3. Pattern 2: Tree of Thoughts (ToT)
Imagine playing Chess. You don't just move; you simulate 3 possible future moves.
ToT asks the LLM to generate **multiple thoughts** per step, evaluate them, and pick the best path.

1.  **Generate**: Option A, Option B, Option C.
2.  **Evaluate**: "Option A is risky. Option B is safe."
3.  **Select**: Go with B.

*Useful for math, coding, and puzzle solving.*

---

## 4. Pattern 3: Reflexion (Self-Correction)
Agents make mistakes.
**Reflexion** adds a "Critic" step.

1.  Agent produces draft.
2.  Critic reviews draft: "You missed the error handling."
3.  Agent re-writes draft using Critic's feedback.

---

## 5. Summary
- **ReAct**: Good for simple queries.
- **Plan-and-Execute**: Good for multi-step projects.
- **Reflexion**: Good for quality control.

**Next Up:** **Multi-Agent Orchestration**—Managing teams of agents.

# Day 141-145: Multi-Agent Orchestration

## 1. Introduction
Single agents are powerful, but **Multi-Agent Systems (MAS)** are the future of enterprise AI.
Instead of one "God Mode" agent doing everything (and failing), we create a **team of specialists**.

---

## 2. Core Concepts
### The "Manager-Worker" Pattern
- **Manager (Orchestrator)**: Breaks down the plan, delegates tasks, reviews output.
- **Workers**: Specialized agents (e.g., `Coder`, `Researcher`, `Reviewer`).

### Communication Styles
1.  **Sequential**: A -> B -> C (Pipeline).
2.  **Hierarchical**: Manager talks for everyone.
3.  **Joint Chat**: Everyone sees every message (like a Slack channel).

---

## 3. Frameworks
### 3.1 AutoGen (Microsoft)
The leading framework for conversational agents.
- **Key Idea**: Agents are just "participants" in a group chat.
- **Code**:
  ```python
  from autogen import AssistantAgent, UserProxyAgent
  
  assistant = AssistantAgent("coder", llm_config=...)
  user = UserProxyAgent("user", execution_config={"work_dir": "coding"})
  
  user.initiate_chat(assistant, message="Plot a chart of NVDA stock price year-to-date.")
  ```
  The `assistant` writes code, the `user` proxy executes it locally.

### 3.2 CrewAI
A layer on top of LangChain to make MAS easy.
- **Agents**: Define Role, Goal, Backstory.
- **Tasks**: Specific description + Expected Output.
- **Process**: Sequential or Hierarchical.

---

## 4. Design Patterns
1.  **Reflexion**: Agent A writes, Agent B critiques, Agent A rewrites.
2.  **Tool Use**: One agent has "Internet Access", another has "File Access". Secure separation of concerns.

---

## 5. Practical Exercise
Build a "Software House":
- **Product Manager**: Writes requirements.
- **Developer**: Writes code.
- **QA Engineer**: Writes tests.
- **Task**: "Create a Snake game in Python."

---

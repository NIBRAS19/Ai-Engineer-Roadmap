# Day 116: LangChain Core Agents

## 1. Introduction
Manually parsing JSON and looping actions is tedious.
**LangChain** abstracts this into the `AgentExecutor`.

---

## 2. Defining Tools
We use the `@tool` decorator.

```python
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]
```

---

## 3. Initializing the Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_FUNCTIONS, 
    verbose=True
)

agent.run("How many letters are in the word 'Antigravity'?")
```

**Output**:
1.  **Thought**: I need to use `get_word_length`.
2.  **Action**: `get_word_length('Antigravity')`.
3.  **Observation**: 11.
4.  **Answer**: 11 letters.

---

## 4. Summary
- **AgentExecutor**: The runtime that manages the loop.
- **Toolkit**: A collection of tools (e.g., GmailToolkit, JiraToolkit).

**Next Up:** **Multi-Agent Systems**—Teams of robots.

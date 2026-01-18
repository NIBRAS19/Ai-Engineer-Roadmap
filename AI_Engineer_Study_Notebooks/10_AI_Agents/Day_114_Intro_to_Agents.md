# Day 114: The Anatomy of an AI Agent

## 1. Beyond the Chatbot
A Chatbot **talks**. An Agent **does**.
- **Chatbot**: "I can tell you how to buy a ticket."
- **Agent**: "I just bought your ticket. It's in your email."

An Agent is an LLM wrapper with access to:
1.  **Brain** (The LLM)
2.  **Hands** (Tools)
3.  **Ears** (Sensors/Environment)
4.  **Memory** (History/State)

---

## 2. The Core Loop: ReAct (Reason + Act)
In 2022, researchers found that asking the LLM to "Think" before "Acting" improved performance massively.

### The Cycle
1.  **Input**: "Who is the CEO of OpenAI?"
2.  **Thought**: "I need to search for the current CEO."
3.  **Action**: `SearchTool("OpenAI CEO current")`
4.  **Observation** (from Tool): "Sam Altman is the CEO..."
5.  **Thought**: "I have the answer."
6.  **Final Answer**: "Sam Altman."

### Error Handling
What if the tool fails?
- **Observation**: "Error: 500 Server Error".
- **Thought**: "The search failed. I should try Wikipedia instead."
*Agents recover from errors. Chains break.*

---

## 3. Tool Schema Design
You must define tools precisely. The LLM is a programmer that reads your function selection.

```python
# Bad Description
def search(query):
    """Searches the web."""

# Good Description (The LLM understands this!)
def search(query):
    """
    Useful for finding current events, facts, or news. 
    Input should be a specific search query.
    Returns a snippet of text from Google.
    """
```
*If your agent is confused, check your tool descriptions/docstrings!*

---

## 4. Memory Systems
An agent handles tasks over time.

### 4.1 Short-Term Memory
- **The Context Window**: Storing the current conversation history (Thoughts, Actions, Observations).
- **Challenge**: Context window fills up.

### 4.2 Long-Term Memory
- **Vector Database (RAG)**: Storing past experiences or huge documents.
- **Reflection**: Summarizing past actions to "learn" strategies.

---

## 5. Summary
- **ReAct**: The standard loop for agent reasoning.
- **Tools**: Functions that the LLM can "call" by generating JSON.
- **Resilience**: Agents can retry actions based on new observations.

**Next Up:** **Function Calling**—The implementation details.


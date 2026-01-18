# Day 115: Function Calling

## 1. OpenAI Function Calling
How does an LLM "click a button"?
We describe the tool to the LLM in JSON format.
The LLM outputs a **JSON object** containing the function name and arguments.

---

## 2. The Workflow
1.  **User**: "What's the weather in Boston?"
2.  **System**: Sends prompt + Function Definition (`get_current_weather(location)`).
3.  **LLM**: Returns JSON `{"function": "get_current_weather", "args": {"location": "Boston"}}`.
    *(Note: The LLM does NOT run the code. It just generates the JSON).*
4.  **System**: Executes the Python function `get_current_weather("Boston")`. Result: "Sunny".
5.  **System**: Sends the result back to the LLM.
6.  **LLM**: "It is sunny in Boston."

---

## 3. Usage (OpenAI API)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Weather in NY?"}],
    tools=tools
)
```

---

## 4. Summary
- **Structured Output**: Function calling forces the LLM to speak JSON.
- **Execution**: You (the developer) are responsible for actually running the tool.

**Next Up:** **LangChain Agents**—The high-level wrapper.

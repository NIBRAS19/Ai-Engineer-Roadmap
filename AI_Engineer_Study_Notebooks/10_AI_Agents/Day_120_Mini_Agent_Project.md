# Day 120: Project - Autonomous Research Agent

## 1. Goal
Build an agent that can:
1.  Take a topic (e.g., "Latest advancements in Fusion Energy").
2.  **Browse the Web** (using Tavily/SerpAPI).
3.  **Read** content.
4.  **Write** a summary report saved as a Markdown file.

---

## 2. Checklist

### Phase 1: Tools
- [ ] Get API Key for **Tavily Search** (optimized for LLMs).
- [ ] Create a `SearchTool` using LangChain.

### Phase 2: The Agent
- [ ] Use `OpenAI Functions` agent.
- [ ] Give it a system prompt: "You are a researcher. Always cite your sources."

### Phase 3: The Interaction
- [ ] User Input: "Research the impact of AI on healthcare in 2024."
- [ ] Agent Loop:
    - Search: "AI healthcare 2024 impact"
    - Read: URL 1, URL 2.
    - Search: "Generative AI in drug discovery" (Follow-up).
- [ ] Final Output: Detailed report.

### Phase 4: UI (Streamlit)
- [ ] Display the "Thought Process" (The agent's log) in the UI so the user sees it thinking.

---

## 3. Bonus Challenge
- **Human-in-the-loop**: Make the agent ask you for clarification if it gets stuck.
- **Memory**: Save past research reports to a Vector DB so it doesn't repeat work.

**CONGRATULATIONS!** You have completed **Weeks 17-18: AI Agents**.
You are now at the cutting edge of AI development.
**Next Week:** **MLOps**—How to deploy and monitor these systems in the real world.

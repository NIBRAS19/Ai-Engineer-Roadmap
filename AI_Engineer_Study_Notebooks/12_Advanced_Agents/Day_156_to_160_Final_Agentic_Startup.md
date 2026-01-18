# Day 156-160: The "Startup" Capstone

## 1. The Final Exam
You have reached the end of the roadmap.
Your task is not to follow a tutorial.
Your task is to build a **Product**.

---

## 2. Project Ideas (Choose One)
### A. The "Junior Dev" Agent (Software Engineering)
- **Input**: A GitHub Issue URL.
- **Action**: Clone repo, reproduce bug, write fix, run tests, open PR.
- **Tech**: AutoGen/CrewAI, Docker sandbox, GitHub API.

### B. The "Market Analyst" Swarm (Finance)
- **Input**: "Should I buy NVIDIA?"
- **Action**:
    - Agent A reads 10-K filings.
    - Agent B scrapes Reddit sentiment.
    - Agent C analyzes technical charts.
    - Manager Agent synthesizes a report.
- **Tech**: LangGraph, Tavily API, Vector DB for history.

### C. The "Personal Recruiter" (HR)
- **Input**: A LinkedIn profile PDF + Job Description.
- **Action**: Conduct a voice interview (Speech-to-Text -> LLM -> Text-to-Speech). Score the candidate.
- **Tech**: OpenAI Realtime API / Deepgram, RAG for resume context.

---

## 3. Requirements
1.  **Multi-Agent**: Must involve at least 2 distinct agents (e.g., Researcher + Writer).
2.  **Tool Use**: Must use at least 1 custom tool (not just search).
3.  **UI**: A clean frontend (Next.js/Streamlit).
4.  **Eval**: A test suite (RAGAS/DeepEval) proving it works.

---

## 4. Graduation
If you build this, you are ready for Senior AI Engineer roles.
You understand the **entire stack**:
- Data (Pandas/SQL)
- Math (Calculus/Stats)
- Models (Transformers/PyTorch)
- Systems (RAG/Agents)
- Production (MLOps/Eval)

**You are the expert now.**
**Good luck.**

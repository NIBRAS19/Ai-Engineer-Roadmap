# Days 126-140: Final Capstone - The "Senior AI Engineer" Project

## 1. The Mission
**Objective**: Build, Deploy, and Monitor an End-to-End GenAI SaaS Application.
This is not a tutorial. This is a test of everything you have learned in the last 140 days.

**The Project Idea**: "Contract-QA" (A RAG-based Legal Assistant)
- **User Story**: A lawyer uploads a PDF contract. They ask: "What are the termination conditions?". The bot answers with citations.

---

## 2. Technical Requirements (The "Must-Haves")

| Component | Requirement | Technology |
|:----------|:------------|:-----------|
| **Frontend** | Clean Chat UI, File Uploader | Streamlit / Next.js |
| **Backend** | API with Async support | FastAPI |
| **AI Engine** | RAG Pipeline (Chunking -> Embedding -> Retrieval) | LangChain / LlamaIndex |
| **Database** | Vector Store + Metadata | Pinecone / Weaviate / FAISS |
| **Ops** | Containerization | Docker |
| **Tracking** | Observability for Traces | LangSmith / Arize Phoenix |

---

## 3. The 15-Day Schedule

### 📅 Phase 1: Core Engine (Days 126-129)
- **Day 126**: **Data Ingestion**. Build a script to parse PDFs, clean text, and chunk smartly (Recursive Splitter).
- **Day 127**: **Vector Pipeline**. Setup Embedding Model (OpenAI/HuggingFace) and upsert chunks to Vector DB.
- **Day 128**: **Retrieval Chain**. Implement the RAG logic. Test "Naive RAG".
- **Day 129**: **Evaluation**. Use `RAGAS` or manual test set to measure Faithfulness and Recall. **Optimize chunk size**.

### 📅 Phase 2: Application Layer (Days 130-133)
- **Day 130**: **FastAPI Backend**. Create `/upload` and `/chat` endpoints.
- **Day 131**: **Session Management**. Ensure chat history is stored (PostgreSQL/Redis) so the bot has memory.
- **Day 132**: **Frontend MVP**. Build the Streamlit UI. Connect it to the API.
- **Day 133**: **Agentic Features**. Give the LLM a "Google Search" tool for fact-checking outside the document.

### 📅 Phase 3: MLOps & Production (Days 134-137)
- **Day 134**: **Dockerization**. Write the `Dockerfile` and `docker-compose.yml` (App + DB).
- **Day 135**: **CI/CD**. Setup GitHub Actions to lint code and run unit tests on push.
- **Day 136**: **Deployment**. Deploy to Railway, Render, or AWS EC2.
- **Day 137**: **Monitoring**. Integrate LangSmith to trace calls. Add User Feedback buttons (👍/👎).

### 📅 Phase 4: Polish & Documentation (Days 138-140)
- **Day 138**: **Optimization**. Implement Caching (Redis) to speed up repeated queries.
- **Day 139**: **Documentation**. Write a `README.md` that explains architecture and how to run it. Record a Loom demo.
- **Day 140**: **Launch**. Post it on LinkedIn/Twitter/Portfolio.

---

## 4. Evaluation Criteria
You pass if:
1.  Users can upload a PDF and chat with it.
2.  The app does not crash on large files.
3.  The Docker container builds successfully.
4.  You have a public GitHub repo with clean code.

---

## 5. Is this the end?
You are now a competent AI Engineer. You can build, deploy, and monitor.
But the industry is moving towards **Autonomous Systems**.

**Do not stop here.**
Proceed to **Month 6: Advanced Agentic Architectures** to master Multi-Agent Orchestration and Cognitive Systems.

**Level Up -> [Day 141](../12_Advanced_Agents/Day_141_to_145_Multi_Agent_Orchestration.md)**


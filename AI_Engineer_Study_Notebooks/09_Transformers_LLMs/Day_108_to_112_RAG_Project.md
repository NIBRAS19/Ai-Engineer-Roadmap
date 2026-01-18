# Days 108-112: Project - RAG Chatbot

## 1. Goal
Build a "Chat with PDF" application.
User uploads a PDF. User asks questions. Bot answers based on PDF.

---

## 2. Tools
- **LangChain**: For orchestration.
- **OpenAI API**: For Embeddings and Generation.
- **FAISS**: For Vector Storage.
- **Streamlit**: For UI.

---

## 3. Checklist

### Phase 1: Ingestion
- [ ] Load PDF using `PyPDFLoader`.
- [ ] Split text using `RecursiveCharacterTextSplitter` (overlap is important!).

### Phase 2: Vector Store
- [ ] Initialize `OpenAIEmbeddings`.
- [ ] Create FAISS index from chunks. `docsearch = FAISS.from_documents(chunks, embeddings)`.

### Phase 3: The Chain
- [ ] Create a `RetrievalQA` chain from LangChain.
- [ ] `chain.run("What is the summary of page 5?")`.

### Phase 4: UI
- [ ] Build a simple Streamlit interface with a File Uploader and a Chat Box.

---

## 4. Bonus Challenge
- Add **Memory**: Make the bot remember previous questions in the conversation (`ConversationBufferMemory`).
- Switch to a **Local LLM** (e.g., Llama 2 via Ollama) to make it free and private.

**CONGRATULATIONS!** You have completed **Weeks 15-16: Transformers & LLMs**.
You are now an **AI Engineer** capable of building GenAI applications.
**Next Week:** **AI Agents**—Giving LLMs tools to take action.

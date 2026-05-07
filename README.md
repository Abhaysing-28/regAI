# RegAI — Multi-Agent Regulatory Reporting Intelligence System

A LangGraph-powered multi-agent system that autonomously monitors regulatory data pipelines, detects anomalies, traces failures, and answers questions from regulatory documents.

## Architecture

```
User Query
    ↓
Supervisor Agent  ──── routes based on intent
    ↓
┌─────────────────────────────────────────┐
│                                         │
▼           ▼              ▼              ▼
Anomaly   Pipeline        RAG Doc      Report
Detector  Debugger        Agent        Summarizer
Agent     Agent                        Agent
```

## Tech Stack
- **LangGraph** — multi-agent orchestration and state management
- **LangChain** — chains, document loaders, retrievers
- **Ollama (Llama 3)** — local open-source LLM inference
- **FAISS** — vector store for regulatory document RAG
- **HuggingFace sentence-transformers** — embeddings (`all-MiniLM-L6-v2`)
- **Streamlit** — interactive demo UI
- **Pandas / NumPy** — rule-based data quality checks

## Setup

### 1. Install Ollama and pull Llama 3
```bash
# Install Ollama: https://ollama.com
ollama pull llama3
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data
```bash
cd data
python generate_synthetic_data.py
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. (Optional) Add real regulatory PDFs
Drop any public regulatory PDFs (Basel III, CSDR) into `data/docs/`.
The RAG agent will auto-index them on first run.

## Agents

| Agent | Responsibility | Input |
|---|---|---|
| Supervisor | Routes queries to correct agent(s) | User query |
| Anomaly Detector | Rule-based + LLM data quality checks | CSV report |
| Pipeline Debugger | Root cause analysis from execution logs | JSON logs |
| RAG Agent | Answers regulatory rule questions | FAISS index |
| Summarizer | Generates executive audit report | All agent outputs |

## Example Queries
- *"Check this report for data quality issues and give me a full audit summary."*
- *"Why did the pipeline fail? Give me root cause and fix steps."*
- *"What are the CSDR Article 9 reporting requirements for null fields?"*
- *"Detect anomalies, debug the pipeline, and summarise all findings."*

## Results
- Detects 6 categories of data anomalies across 100+ simulated regulatory reports
- Traces pipeline failures to root cause stage with specific error context
- Answers regulatory compliance questions using RAG over CSDR/Basel III documents
- Reduces multi-step manual audit review to a single natural language query

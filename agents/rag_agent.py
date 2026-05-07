"""
RAG Document Agent
Indexes regulatory PDF documents using FAISS + HuggingFace embeddings.
Answers natural language questions about regulatory rules and requirements.
"""
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.state import AgentState

VECTORSTORE_PATH = "vectorstore/regulatory_index"
DOCS_PATH        = "data/docs"

# Fallback synthetic regulatory text if no PDFs are present
SYNTHETIC_REG_TEXT = """
CSDR Article 9 — Settlement Discipline
Under CSDR Article 9, investment firms must report details of transactions in financial instruments
to the competent authority. Reports must be submitted no later than the close of the following
working day. Reports must include: trade date, settlement date, ISIN, quantity, price, counterparty ID,
trading venue, and jurisdiction of the counterparty.

Null values in mandatory fields (counterparty_id, trade_id, notional_amount) will result in
rejection of the report. Duplicate trade_id values are not permitted within a single reporting period.
Negative notional amounts are invalid and must be flagged for manual review.

CSDR Article 7 — Settlement Fails
Settlement fails must be reported within 2 business days of the intended settlement date.
Penalties apply at a daily rate based on the asset class and transaction size.
Cash penalties are calculated daily until the settlement fail is resolved.

Basel III — Capital Requirements
Banks must maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5%.
The capital conservation buffer requirement is an additional 2.5% of risk-weighted assets.
Leverage ratio must be maintained at a minimum of 3%.
Liquidity Coverage Ratio (LCR) must be at least 100%.

NCCBR — Non-Cleared Bilateral Reporting
All non-centrally cleared OTC derivatives must be reported to a trade repository.
Reports must include: notional amount, currency, maturity date, counterparty LEI codes,
collateral posted, and margin requirements. Reports are due T+1.
"""


def build_or_load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load existing index if available
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Build index from PDFs if available, else use synthetic text
    docs = []
    os.makedirs(DOCS_PATH, exist_ok=True)

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    if pdf_files:
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(DOCS_PATH, pdf))
            docs.extend(loader.load())
    else:
        # Write synthetic regulatory text as fallback
        fallback_path = os.path.join(DOCS_PATH, "regulatory_rules.txt")
        with open(fallback_path, "w") as f:
            f.write(SYNTHETIC_REG_TEXT)
        loader = TextLoader(fallback_path)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore


def rag_agent_node(state: AgentState) -> AgentState:
    query = state["user_query"]

    vectorstore = build_or_load_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs        = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )
    prompt = f"""You are a regulatory compliance expert with deep knowledge of CSDR, Basel III, and NCCBR reporting rules.

Use the context below to answer the question. Be specific and cite rule names where relevant.
If the answer is not in the context, say so clearly — do not hallucinate.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, structured answer."""

    response = llm.invoke([HumanMessage(content=prompt)])
    rag_answer = response.content.strip()

    agents_called = state.get("agents_called", []) + ["rag_agent"]
    return {**state, "rag_answer": rag_answer, "agents_called": agents_called}

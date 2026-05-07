"""
RegAI — Streamlit UI
Multi-Agent Regulatory Reporting Intelligence System
"""
import streamlit as st
import sys
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RegAI")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

sys.path.insert(0, os.path.dirname(__file__))

from graph import build_graph

st.set_page_config(
    page_title="RegAI — Regulatory Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextArea textarea { font-family: monospace; font-size: 13px; }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-anomaly  { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b55; }
    .badge-debug    { background: #ffa50022; color: #ffa500; border: 1px solid #ffa50055; }
    .badge-rag      { background: #00c8ff22; color: #00c8ff; border: 1px solid #00c8ff55; }
    .badge-summary  { background: #00ff8822; color: #00ff88; border: 1px solid #00ff8855; }
    .badge-comparator { background: #9b59b622; color: #9b59b6; border: 1px solid #9b59b655; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏦 RegAI — Regulatory Reporting Intelligence")
st.caption("Multi-agent system for anomaly detection, pipeline debugging, and regulatory Q&A")
st.divider()

# ── Sidebar — Inputs ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Radio toggle for data source
    data_mode = st.radio(
        "Data Source",
        ["Use sample data", "Upload your own"],
        help="Choose whether to use pre-loaded sample data or upload your own files"
    )

    if data_mode == "Use sample data":
        report_path = st.selectbox(
            "Regulatory Report",
            ["data/reports/csdr_report_anomalous.csv", "data/reports/csdr_report_clean.csv"],
            help="Select a synthetic regulatory report CSV"
        )

        log_path = st.selectbox(
            "Pipeline Log",
            ["data/logs/pipeline_failure.json", "data/logs/pipeline_success.json"],
            help="Select a pipeline execution log"
        )
    else:  # Upload your own
        # Create tmp directory if it doesn't exist
        import tempfile

        tmp_dir = tempfile.mkdtemp()

        # Upload regulatory report
        report_file = st.file_uploader("Upload Regulatory Report", type=["csv"])
        report_path = None
        if report_file is not None:
            report_path = os.path.join(tmp_dir, report_file.name)
            with open(report_path, "wb") as f:
                f.write(report_file.getbuffer())
            # Load and show row count
            try:
                df = pd.read_csv(report_path)
                st.success(f"Loaded {report_file.name} — {len(df)} rows")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

        # Upload pipeline log
        log_file = st.file_uploader("Upload Pipeline Log", type=["json"])
        log_path = None
        if log_file is not None:
            log_path = os.path.join(tmp_dir, log_file.name)
            with open(log_path, "wb") as f:
                f.write(log_file.getbuffer())

        # Upload comparison report (optional)
        compare_file = st.file_uploader("Upload Comparison Report (optional)", type=["csv"])
        report_path_compare = None
        if compare_file is not None:
            report_path_compare = os.path.join(tmp_dir, compare_file.name)
            with open(report_path_compare, "wb") as f:
                f.write(compare_file.getbuffer())

    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "Check this report for data quality issues and give me a full audit summary.",
        "Why did the pipeline fail? Give me root cause and fix steps.",
        "What are the CSDR Article 9 reporting requirements for null fields?",
        "Detect anomalies, debug pipeline failures, and summarise all findings.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex[:20]):
            st.session_state["query_input"] = ex

    st.divider()
    st.caption("Model: GPT-OSS-120B (Free) via OpenRouter\nEmbeddings: all-MiniLM-L6-v2")

    # LangSmith tracing status
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        st.success("🟢 LangSmith tracing active")
    else:
        st.caption("⚪ LangSmith tracing disabled")

# ── Main — Query Input ─────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_area(
        "Enter your query",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g. Check this report for anomalies and debug any pipeline failures, then summarise..."
    )
with col2:
    st.write("")
    st.write("")
    run = st.button("▶ Run Agents", use_container_width=True, type="primary")

# ── Main — Agent Execution ─────────────────────────────────────────────────
if run and query.strip():
    with st.spinner("Initialising agent graph..."):
        graph = build_graph()

    initial_state = {
        "user_query": query,
        "report_path": report_path,
        "log_path": log_path,
        "report_path_compare": report_path_compare if 'report_path_compare' in locals() else None,
        "messages": [],
        "agents_called": [],
        "anomaly_report": None,
        "debug_report": None,
        "rag_answer": None,
        "final_summary": None,
        "comparison_report": None,
        "next_agent": None,
    }

    progress = st.empty()
    result = {}

    with st.spinner("Agents working..."):
        # Stream graph execution with higher recursion limit to prevent GraphRecursionError
        # This allows the supervisor to route through multiple agents without hitting the limit
        config = {"recursion_limit": 100}
        for step in graph.stream(initial_state, config=config):
            node_name = list(step.keys())[0]
            state = list(step.values())[0]
            if node_name != "supervisor":
                progress.info(f"⚡ Agent running: **{node_name.replace('_', ' ').title()}**")
            result = state

progress.empty()

# ── Results Display ────────────────────────────────────────────────────
agents_called = result.get("agents_called", [])

# Agent badges
badge_map = {
    "anomaly_detector": ("badge-anomaly", "🔍 Anomaly Detector"),
    "pipeline_debugger": ("badge-debug", "🐛 Pipeline Debugger"),
    "rag_agent": ("badge-rag", "📚 RAG Agent"),
    "report_comparator": ("badge-comparator", "🔄 Report Comparator"),
    "summarizer": ("badge-summary", "📋 Summarizer"),
}
badge_html = " ".join([
    f'<span class="agent-badge {badge_map[a][0]}">{badge_map[a][1]}</span>'
    for a in agents_called if a in badge_map
])
st.markdown(f"**Agents invoked:** {badge_html}", unsafe_allow_html=True)
st.divider()

# Tab display for each agent output
tabs = []
tab_labels = []
if result.get("anomaly_report"):
    tab_labels.append("🔍 Anomaly Report")
if result.get("debug_report"):
    tab_labels.append("🐛 Pipeline Debug")
if result.get("rag_answer"):
    tab_labels.append("📚 Regulatory Q&A")
if result.get("comparison_report"):
    tab_labels.append("🔄 Comparison Report")
if result.get("final_summary"):
    tab_labels.append("📋 Audit Summary")

if tab_labels:
    tabs = st.tabs(tab_labels)
    idx = 0

    if result.get("anomaly_report"):
        with tabs[idx]:
            st.markdown(result["anomaly_report"])
        idx += 1

    if result.get("debug_report"):
        with tabs[idx]:
            st.markdown(result["debug_report"])
        idx += 1

    if result.get("rag_answer"):
        with tabs[idx]:
            st.markdown(result["rag_answer"])
        idx += 1

    if result.get("comparison_report"):
        with tabs[idx]:
            st.markdown(result["comparison_report"])
        idx += 1

    if result.get("final_summary"):
        with tabs[idx]:
            st.markdown(result["final_summary"])
            st.download_button(
                "⬇ Download Audit Report",
                data=result["final_summary"],
                file_name="regai_audit_report.md",
                mime="text/markdown")
    else:
        st.warning("No agent outputs generated. Check that your OPENROUTER_API_KEY is set in .env")

elif run and not query.strip():
    st.warning("Please enter a query before running.")

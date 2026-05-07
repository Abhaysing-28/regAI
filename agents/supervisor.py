"""
Supervisor Agent — routes user queries to the correct agent(s).
Uses Ollama LLM to decide routing based on user intent.
"""
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.state import AgentState


ROUTING_PROMPT = """You are a supervisor for a regulatory reporting AI system.
Your job is to route the user query to the correct agent(s).

Available agents:
- anomaly_detector : Detects data anomalies, duplicates, nulls, schema issues in regulatory report CSV files
- pipeline_debugger: Traces and explains pipeline failures from log files
- rag_agent        : Answers questions about regulatory documents (CSDR, Basel III rules etc.)
- report_comparator: Compares two regulatory report CSV files, detecting schema changes, value drifts, new/removed records, and data quality regressions between reporting periods
- summarizer       : Generates final audit summary combining all findings
- FINISH           : All required agents have been called, return final answer

Already called agents: {agents_called}

User query: {query}

Respond with ONLY the name of the next agent to call, or FINISH.
No explanation. Just the agent name."""


def supervisor_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )

    agents_called = state.get("agents_called", [])

    # If all relevant agents called or summarizer done, finish
    if "summarizer" in agents_called:
        return {**state, "next_agent": "FINISH"}

    prompt = ROUTING_PROMPT.format(
        agents_called=", ".join(agents_called) if agents_called else "none",
        query=state["user_query"]
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_query"])
    ])

    next_agent = response.content.strip().lower().replace("-", "_")

    # Validate response
    valid_agents = ["anomaly_detector", "pipeline_debugger", "rag_agent", "report_comparator", "summarizer", "finish"]
    if next_agent not in valid_agents:
        next_agent = "summarizer"

    return {**state, "next_agent": next_agent}


def route_next(state: AgentState) -> str:
    """LangGraph conditional edge function."""
    return state.get("next_agent", "FINISH")

"""
Report Summarizer Agent
Combines outputs from all other agents into a
structured executive audit summary.
"""
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.state import AgentState


def summarizer_node(state: AgentState) -> AgentState:
    sections = []

    if state.get("anomaly_report"):
        sections.append(f"=== ANOMALY DETECTION FINDINGS ===\n{state['anomaly_report']}")

    if state.get("debug_report"):
        sections.append(f"=== PIPELINE DEBUG ANALYSIS ===\n{state['debug_report']}")

    if state.get("rag_answer"):
        sections.append(f"=== REGULATORY DOCUMENT Q&A ===\n{state['rag_answer']}")

    if not sections:
        return {
            **state,
            "final_summary": "No agent outputs available to summarize.",
            "agents_called": state.get("agents_called", []) + ["summarizer"]
        }

    combined = "\n\n".join(sections)

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )
    prompt = f"""You are a chief data officer preparing an executive audit report for a regulatory reporting system.

Synthesize the following agent findings into a single, cohesive audit summary report.

Structure your report as:
# RegAI Audit Summary Report

## Executive Overview
(2-3 sentences summarising overall system health and key findings)

## Key Findings
(Bullet points of the most critical issues across all agents)

## Risk Assessment
(HIGH / MEDIUM / LOW rating with brief justification)

## Recommended Next Steps
(Prioritised action list)

## Agent Reports Referenced
(List which agents contributed findings)

---
AGENT FINDINGS:
{combined}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    final_summary = response.content.strip()

    agents_called = state.get("agents_called", []) + ["summarizer"]
    return {**state, "final_summary": final_summary, "agents_called": agents_called}

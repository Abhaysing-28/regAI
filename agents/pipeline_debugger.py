"""
Pipeline Debugger Agent
Parses pipeline execution logs, identifies failure points,
and uses LLM to produce root cause analysis and fix suggestions.
"""
import json
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.state import AgentState


def parse_logs(log_path: str) -> tuple[list, list]:
    """Parse log file and separate successful and failed stages."""
    with open(log_path, "r") as f:
        logs = json.load(f)

    success = [l for l in logs if l.get("status") == "SUCCESS"]
    failed  = [l for l in logs if l.get("status") == "FAILED"]
    return success, failed


def pipeline_debugger_node(state: AgentState) -> AgentState:
    log_path = state.get("log_path", "data/logs/pipeline_failure.json")

    try:
        success_stages, failed_stages = parse_logs(log_path)
    except Exception as e:
        return {
            **state,
            "debug_report": f"Error reading logs: {e}",
            "agents_called": state.get("agents_called", []) + ["pipeline_debugger"]
        }

    # Build structured log summary for LLM
    success_summary = "\n".join([
        f"  ✓ [{l['stage']}] {l['duration_sec']}s — {l['records_in']} records in, {l['records_out']} out"
        for l in success_stages
    ])

    if failed_stages:
        failure_detail = "\n".join([
            f"  ✗ [{l['stage']}] ERROR {l.get('error_code','UNKNOWN')}\n"
            f"    Message: {l.get('message','')}\n"
            f"    Records in: {l.get('records_in',0)}, Records out: {l.get('records_out',0)}\n"
            f"    Stack: {l.get('stack_trace','N/A')}"
            for l in failed_stages
        ])
        status_line = f"PIPELINE FAILED at stage: {failed_stages[0]['stage']}"
    else:
        failure_detail = "No failures detected."
        status_line = "PIPELINE COMPLETED SUCCESSFULLY"

    log_context = f"""
PIPELINE ID: {success_stages[0]['pipeline_id'] if success_stages else 'UNKNOWN'}
STATUS: {status_line}

SUCCESSFUL STAGES:
{success_summary if success_summary else 'None'}

FAILED STAGES:
{failure_detail}
""".strip()

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )
    prompt = f"""You are a senior data pipeline engineer specialising in regulatory reporting systems.

Analyse the following pipeline execution log and provide:
1. Root Cause Analysis — what exactly went wrong and why
2. Impact Assessment — which downstream stages were affected and what data was lost
3. Step-by-Step Fix — concrete actions to resolve the issue (reference specific column names, error codes)
4. Prevention — how to prevent this failure in future runs

Pipeline Log:
{log_context}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    debug_report = response.content.strip()

    agents_called = state.get("agents_called", []) + ["pipeline_debugger"]
    return {**state, "debug_report": debug_report, "agents_called": agents_called}

"""
Anomaly Detector Agent
Scans regulatory report CSVs for data quality issues using rule-based checks
then passes findings to LLM for structured natural language output.
"""
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.state import AgentState

VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SGD"}
VALID_STATUSES   = {"SETTLED", "PENDING", "FAILED", "CANCELLED"}


def run_rule_checks(df: pd.DataFrame) -> list[dict]:
    """Run deterministic rule-based anomaly checks."""
    findings = []

    # 1. Null checks on critical columns
    critical_cols = ["trade_id", "counterparty_id", "notional_amount", "currency", "status"]
    for col in critical_cols:
        if col in df.columns:
            null_rows = df[df[col].isnull()].index.tolist()
            if null_rows:
                findings.append({
                    "type": "NULL_VALUES",
                    "severity": "HIGH",
                    "column": col,
                    "affected_rows": null_rows[:10],
                    "count": len(null_rows),
                    "description": f"Column '{col}' has {len(null_rows)} null values."
                })

    # 2. Duplicate trade IDs
    if "trade_id" in df.columns:
        dupes = df[df.duplicated("trade_id", keep=False)]["trade_id"].unique().tolist()
        if dupes:
            findings.append({
                "type": "DUPLICATE_IDS",
                "severity": "HIGH",
                "column": "trade_id",
                "affected_values": dupes[:5],
                "count": len(dupes),
                "description": f"Found {len(dupes)} duplicate trade_id values."
            })

    # 3. Negative notional amounts
    if "notional_amount" in df.columns:
        neg_rows = df[df["notional_amount"] < 0].index.tolist()
        if neg_rows:
            findings.append({
                "type": "NEGATIVE_NOTIONAL",
                "severity": "HIGH",
                "column": "notional_amount",
                "affected_rows": neg_rows,
                "count": len(neg_rows),
                "description": f"Found {len(neg_rows)} rows with negative notional amounts."
            })

    # 4. Invalid currency codes
    if "currency" in df.columns:
        invalid_ccy = df[~df["currency"].isin(VALID_CURRENCIES)]["currency"].unique().tolist()
        if invalid_ccy:
            findings.append({
                "type": "INVALID_CURRENCY",
                "severity": "MEDIUM",
                "column": "currency",
                "affected_values": invalid_ccy,
                "count": len(invalid_ccy),
                "description": f"Invalid currency codes found: {invalid_ccy}"
            })

    # 5. Outlier detection (>3 std devs)
    if "notional_amount" in df.columns:
        clean = df["notional_amount"].dropna()
        mean, std = clean.mean(), clean.std()
        outliers = df[np.abs(df["notional_amount"] - mean) > 3 * std].index.tolist()
        if outliers:
            findings.append({
                "type": "STATISTICAL_OUTLIER",
                "severity": "MEDIUM",
                "column": "notional_amount",
                "affected_rows": outliers[:5],
                "count": len(outliers),
                "description": f"Found {len(outliers)} statistical outliers (>3 std devs) in notional_amount."
            })

    return findings


def anomaly_detector_node(state: AgentState) -> AgentState:
    report_path = state.get("report_path", "data/reports/csdr_report_anomalous.csv")

    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        return {
            **state,
            "anomaly_report": f"Error loading report: {e}",
            "agents_called": state.get("agents_called", []) + ["anomaly_detector"]
        }

    findings = run_rule_checks(df)

    if not findings:
        findings_text = "No anomalies detected. Report appears clean."
    else:
        findings_text = "\n".join([
            f"[{f['severity']}] {f['type']} — {f['description']}"
            for f in findings
        ])

    # Pass findings to LLM for structured natural language report
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )
    prompt = f"""You are a regulatory data quality expert.
Below are automated anomaly check results for a CSDR regulatory report with {len(df)} records.

FINDINGS:
{findings_text}

Write a concise, professional anomaly report with:
1. Executive Summary (2 sentences)
2. Critical Issues (HIGH severity)
3. Warnings (MEDIUM severity)
4. Recommended Actions

Be specific and reference column names and row counts."""

    response = llm.invoke([HumanMessage(content=prompt)])
    anomaly_report = response.content.strip()

    agents_called = state.get("agents_called", []) + ["anomaly_detector"]
    return {**state, "anomaly_report": anomaly_report, "agents_called": agents_called}

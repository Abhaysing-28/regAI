"""
Report Comparator Agent
Compares two regulatory report CSV files, detecting schema changes,
value drifts, new/removed records, and data quality regressions
between reporting periods.
"""
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.state import AgentState


def report_comparator_node(state: AgentState) -> AgentState:
    baseline_path = state.get("report_path")
    compare_path = state.get("report_path_compare")

    if not baseline_path or not compare_path:
        return {
            **state,
            "comparison_report": "Error: Both baseline and comparison reports are required.",
            "agents_called": state.get("agents_called", []) + ["report_comparator"]
        }

    try:
        df_baseline = pd.read_csv(baseline_path)
        df_compare = pd.read_csv(compare_path)
    except Exception as e:
        return {
            **state,
            "comparison_report": f"Error loading reports: {e}",
            "agents_called": state.get("agents_called", []) + ["report_comparator"]
        }

    findings = []

    # 1. Schema diff — columns present in one but missing in the other
    baseline_cols = set(df_baseline.columns)
    compare_cols = set(df_compare.columns)
    missing_in_compare = baseline_cols - compare_cols
    missing_in_baseline = compare_cols - baseline_cols

    if missing_in_compare:
        findings.append(f"Columns missing in new report: {', '.join(missing_in_compare)}")
    if missing_in_baseline:
        findings.append(f"New columns in new report: {', '.join(missing_in_baseline)}")

    # 2. Row count diff — difference in total record counts
    baseline_count = len(df_baseline)
    compare_count = len(df_compare)
    count_diff = compare_count - baseline_count
    findings.append(f"Row count change: {baseline_count} → {compare_count} ({count_diff:+d} records)")

    # 3. Duplicate trade_ids that appear in both reports
    if "trade_id" in df_baseline.columns and "trade_id" in df_compare.columns:
        baseline_ids = set(df_baseline["trade_id"].dropna())
        compare_ids = set(df_compare["trade_id"].dropna())
        common_ids = baseline_ids & compare_ids
        if common_ids:
            findings.append(f"Common trade_ids in both reports: {len(common_ids)} records")

    # 4. Value changes — for matching trade_ids, flag rows where notional_amount, status, or currency changed
    value_changes = []
    if "trade_id" in df_baseline.columns and "trade_id" in df_compare.columns:
        df_baseline_indexed = df_baseline.set_index("trade_id")
        df_compare_indexed = df_compare.set_index("trade_id")

        for trade_id in common_ids:
            if trade_id in df_baseline_indexed.index and trade_id in df_compare_indexed.index:
                baseline_row = df_baseline_indexed.loc[trade_id]
                compare_row = df_compare_indexed.loc[trade_id]

                changes = []
                for col in ["notional_amount", "status", "currency"]:
                    if col in baseline_row.index and col in compare_row.index:
                        if baseline_row[col] != compare_row[col]:
                            changes.append(f"{col}: {baseline_row[col]} → {compare_row[col]}")

                if changes:
                    value_changes.append(f"Trade ID {trade_id}: {', '.join(changes)}")

    if value_changes:
        findings.append(f"Value changes in matching records: {len(value_changes)} records affected")
        findings.extend(value_changes[:5])  # Show first 5 examples

    # 5. New records — trade_ids present in new report but not baseline
    new_ids = compare_ids - baseline_ids
    if new_ids:
        findings.append(f"New records in new report: {len(new_ids)} trade_ids")

    # 6. Removed records — trade_ids in baseline but missing from new report
    removed_ids = baseline_ids - compare_ids
    if removed_ids:
        findings.append(f"Removed records from baseline: {len(removed_ids)} trade_ids")

    findings_text = "\n".join(findings) if findings else "No differences detected between reports."

    # Pass findings to LLM for structured comparison report
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )
    prompt = f"""You are a regulatory data analyst specializing in report comparison and data drift detection.

Below are automated comparison results between two CSDR regulatory reports:
- Baseline report: {baseline_count} records
- New report: {compare_count} records

COMPARISON FINDINGS:
{findings_text}

Write a professional comparison report covering:
1. Summary of Changes (2-3 sentences)
2. Critical Differences (schema changes, record removals, value changes)
3. Data Drift Indicators (row count changes, new/removed records)
4. Recommended Review Actions (prioritized by risk level)

Be specific and reference column names, trade_ids, and counts where relevant."""

    response = llm.invoke([HumanMessage(content=prompt)])
    comparison_report = response.content.strip()

    agents_called = state.get("agents_called", []) + ["report_comparator"]
    return {**state, "comparison_report": comparison_report, "agents_called": agents_called}

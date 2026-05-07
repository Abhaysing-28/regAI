"""
LangGraph StateGraph — wires all agents together.
Supervisor routes between agents based on user query.
"""
from langgraph.graph import StateGraph, END
from utils.state import AgentState
from agents.supervisor import supervisor_node, route_next
from agents.anomaly_detector import anomaly_detector_node
from agents.pipeline_debugger import pipeline_debugger_node
from agents.rag_agent import rag_agent_node
from agents.summarizer import summarizer_node
from agents.report_comparator import report_comparator_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("supervisor",        supervisor_node)
    graph.add_node("anomaly_detector",  anomaly_detector_node)
    graph.add_node("pipeline_debugger", pipeline_debugger_node)
    graph.add_node("rag_agent",         rag_agent_node)
    graph.add_node("summarizer",        summarizer_node)
    graph.add_node("report_comparator", report_comparator_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to agents via conditional edges
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "anomaly_detector":  "anomaly_detector",
            "pipeline_debugger": "pipeline_debugger",
            "rag_agent":         "rag_agent",
            "summarizer":        "summarizer",
            "report_comparator": "report_comparator",
            "FINISH":            END,
            "finish":            END,
        }
    )

    # All agents report back to supervisor after completing
    for agent in ["anomaly_detector", "pipeline_debugger", "rag_agent", "summarizer", "report_comparator"]:
        graph.add_edge(agent, "supervisor")

    return graph.compile()


# Quick CLI test
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "user_query": "Check the regulatory report for anomalies and debug any pipeline failures, then give me a full summary.",
        "report_path": "data/reports/csdr_report_anomalous.csv",
        "log_path":    "data/logs/pipeline_failure.json",
        "report_path_compare": None,
        "messages":    [],
        "agents_called": [],
        "anomaly_report": None,
        "debug_report":   None,
        "rag_answer":     None,
        "final_summary":  None,
        "comparison_report": None,
        "next_agent":     None,
    })

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(result.get("final_summary", "No summary generated."))

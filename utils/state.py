"""
Shared LangGraph state definition.
All agents read from and write to this state object.
"""
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]

    # Input context
    user_query: str
    report_path: Optional[str]
    log_path: Optional[str]
    report_path_compare: Optional[str]

    # Agent outputs
    anomaly_report: Optional[str]
    debug_report: Optional[str]
    rag_answer: Optional[str]
    final_summary: Optional[str]
    comparison_report: Optional[str]

    # Routing
    next_agent: Optional[str]
    agents_called: List[str]
